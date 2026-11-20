# Learnings

One section per module. Keep notes in your own words — what clicked, what
surprised you, and the one-line definition of each new term.

## Module 0 — Skeleton

## Module 1

- **The model decides, the loop just drives.** `run_agent` is a `while` loop: send the conversation to Claude, if `stop == "tool"` run the tools and append their results as a user turn, repeat until `stop == "end"` or we hit `MAX_STEPS`. The agent itself never chooses a tool — it only runs the one Claude asked for.
- **Tool results go back as a fresh user turn.** Each `ToolResult` carries the `id` of the `ToolCall` it answers; Claude pairs them up. Fail-safe errors (bad input, port exception) come back as `ToolResult(is_error=True, content="<type>: <msg>")` — the model sees the error and can recover, instead of the loop crashing.
- **`stop_reason` is the only "are we done?" signal.** `"end_turn"` → final answer. `"tool_use"` → there are tool calls waiting; running them and looping is my job, not Claude's.
- **Cost = dot product of tokens and a price table.** For every call we read `usage.input_tokens / output_tokens / cache_read_input_tokens / cache_creation_input_tokens`, multiply each by its Haiku price ($/M tokens), sum, divide by 1e6. One log line per call means `/usage` later is just a `SELECT sum(cost_usd)`.
- **Neutral types are the firewall.** The agent only imports from `dobby.llm.types` and `dobby.llm.base`. Swapping Haiku for Ollama (Module 2) means writing one new adapter — zero changes to `agent.py` or the tests. The round-trip conversion test is what makes that swap safe.
- **One production risk I'd watch.** The system prompt tells Claude "call `list_events` once before any `create_event`", but it's a soft rule — a confused model could still skip it and create a duplicate. Module 2's eval `tool_trajectory` grader is what turns this prompt-level rule into a measured one.

**Measured cost vs PRD estimate (2026-11-17):** 3 PRD messages on Haiku 4.5, empty calendar, no prompt caching. Average **$0.000467/msg** (42% under the $0.0008 estimate). Call counts: 3, 3, 1.

## Module 2

**Part A — the eval harness (Claude only).**

- **Eval vs test.** A unit test says "this function returns 42"; an eval says "across 30 pasted messages, dates are correct ≥ 95% of the time." Golden set in `evals/cases.jsonl`, code graders in `evals/graders.py`, runner in `evals/run_evals.py`, one command: `make eval MODEL=... PROVIDER=... RUNS=3`.
- **Six code graders** (no LLM judge yet): date/time accuracy, event recall, event precision, duplicate avoidance (for `duplicate` cases only), tool trajectory (list_events before create_event?), injection safety (for `injection` cases only). Grader tests in `tests/test_graders.py` — 26 unit tests pin them down so a wobble in the measuring instrument doesn't masquerade as a model regression.
- **Why N = 3 and "mean score".** LLMs are non-deterministic even on identical input. One run per case can't tell "the prompt got better" from "lucky dice." PRD prescribes N=3 and reporting the mean.
- **Claude 4.x surprise: `temperature` has been removed from Anthropic's v2 Messages API** (SDK ≥ 2.x). `messages.create(temperature=0)` now returns `"temperature: Extra inputs are not permitted"`. The v2 API replaced it with `output_config.effort` (`low`/`medium`/`high`/`xhigh`/`max`). So our "temperature=0" determinism lever doesn't apply to Haiku 4.5. We keep the kwarg on `AnthropicClient` for signature uniformity across adapters (Ollama still uses temperature), accept the native variance, and lean on N=3 averaging.
- **Haiku baseline (30 cases × 3 runs, 2026-11-17).** 5 of 6 thresholds passed: date_time 97.8%, recall 98.9%, precision 96.9%, tool_trajectory 100%, injection_safety 100%. Only miss: duplicate_avoidance at 88.9% vs 100% threshold — all 3 failing cases are one-flake-in-three. Cost $0.0421 total, 5m17s.
- **Injection safety holding at 100%** on three adversarial cases (fake SYSTEM OVERRIDE, fake tool-name instructions, fake "debug mode") is the single most reassuring eval. The two-tool allowlist + "treat text as data" system-prompt rule is doing real work.

**Part B, the Ollama adapter.**

- **One protocol, two shapes.** Anthropic's Messages API bundles tool results into a single user turn with content blocks. OpenAI's chat completions API (what Ollama speaks on `/v1`) wants one `role: "tool"` message per result and stringifies tool-call arguments as JSON. The neutral `Message` and `ToolCall` types kept the agent code unchanged. Only the two adapters know those differences exist.
- **No new dependencies.** The adapter uses stdlib `urllib.request` for a single JSON POST. The `openai` SDK and `httpx` would both work, but adding a package just to call one endpoint is overkill. Keep it small until it hurts.
- **Small-model quirks worth catching in the adapter.** Some Ollama models return `arguments` as a dict instead of a JSON string. Some emit `tool_calls` alongside `finish_reason: "stop"` (should be `"tool_calls"`). Both are handled: the adapter normalises to the neutral `ToolCall` and treats presence-of-tool-calls as "we need to run them."
- **Preflight check.** Before an Ollama eval kicks off, the runner hits `/api/tags` to confirm the model is pulled. One `SystemExit` with a clear "run `ollama pull <model>`" is better than ninety timeouts an hour in.

**Part C, head-to-head comparison.**

- **Haiku 4.5 is the production candidate.** 5 of 6 thresholds green: date 97.8%, recall 98.9%, precision 96.9%, tool trajectory 100%, injection safety 100%. Only miss is duplicate avoidance at 88.9%, three one-in-three flakes.
- **Qwen3:8b is not viable yet on the current prompt.** 1 of 6 thresholds green. Date accuracy 86.7%, recall 90.0%, precision 86.1%, tool trajectory 94.4%, injection safety 33.3%. Only pass is duplicate avoidance at 100%, Qwen was more mechanical and didn't flake.
- **The hard blocker is injection safety.** Qwen followed a fake "SYSTEM OVERRIDE" in `injection-01` (three timeouts, model spun) and a fake "AGENT INSTRUCTION" in `injection-03` (created phantom events every run, precision 0.50). This alone makes it unshippable for v1. The two gaps on dates and tool trajectory are fixable with prompt work; the gap on injection is a safety-property gap and would need either prompt hardening with a second eval pass, or a bigger open model.
- **Cost and latency.** Haiku $0.000468/msg, p95 around 5.5s. Qwen $0, p95 around 90s on M-series Mac (6x slower).
- **The dashboard is the output.** `docs/index.html` is the humanised PRD and the live eval scorecard in one page. `scripts/update_dashboard.py` regenerates it from `evals/results/*.json` so the scorecard stays truthful on every run. Served via GitHub Pages at the repo root.

## Module 3

- **Two files, two jobs.** `credentials.json` says "this is the Dobby app" (the Cloud Console OAuth client); `token.json` says "this is Aruna, and she said yes." The first is static and you downloaded it once; the second is written by the library after the browser flow and silently refreshed from then on. Lose `token.json` and Dobby just re-opens the browser next run. Lose `credentials.json` and the app is gone.
- **Adapter = translation layer.** `GoogleCalendar` implements `CalendarPort`, so `agent.py`, the tests, and the evals didn't change a line. `_to_google` and `_from_google` are the two ~20-line pure functions that convert between our clean `Event` dataclass and Google's nested JSON — tested in isolation with fixtures, no network. Every external service in Dobby will follow this shape.
- **`extendedProperties.private` is the clean tag.** Google Calendar's built-in key/value metadata on events is invisible in the UI and queryable via API, so `dobby=1` doesn't pollute notes/description. Later modules (`/undo`, `/usage`) will filter by this tag to find "events Dobby created."
- **All-day event end dates are exclusive.** Google wants `end.date = start.date + 1 day` for a one-day event, and `end_date + 1` for multi-day. The agent's `Event` model uses inclusive end_date, so the adapter adds 1 on write and subtracts 1 on read. Got this wrong once; fixtures caught it.
- **One macOS gotcha caught us twice now.** `uv run` (without `--no-sync`) re-hides the editable `.pth` file every call, so `python -m dobby` fails with `ModuleNotFoundError`. Fix is a `make dobby ARGS='...'` target that does `unhide` first then `uv run --no-sync`. Don't trust that `make unhide` alone is enough if the next command re-syncs.

## Module 4

## Module 4b

## Module 5

## Module 6

## Module 7

## Module 8

## Module 9
