# dobby-agent

Dobby turns pasted scheduling messages into calendar events. It works for
anyone who gets dates buried in WhatsApp groups, email chains, forwarded
school circulars, or newsletters. Paste once. Reminders arrive a week
ahead, two days ahead, and on the day itself. Events can stay private or
go to a shared family calendar.

Built in the open as a product manager portfolio project.

**[Open the live dashboard](https://arunasivaram.github.io/Dobbyagent/)**. PRD, eval scorecard, cost comparison, latency, and build progress in one page.

Raw PRD in [`docs/PRD.md`](docs/PRD.md). Per module reflection notes in [`LEARNINGS.md`](LEARNINGS.md).

## Latest evals

<!-- BEGIN:scorecard -->
_Latest evals · updated 2026-11-25 15:08 UTC_

| Model | Hosting | Cost / msg | Latency p95 | Thresholds passed | Status |
| --- | --- | --- | --- | --- | --- |
| `claude-haiku-4-5-20251001` | anthropic | $0.000568 | 11,887 ms | 6 / 6 | measured |
| `qwen3:8b` | ollama | $0.000000 | 74,867 ms | 1 / 6 | measured |

**Per-metric scores:**

| Metric | Threshold | `claude-haiku-4-5-20251001` | `qwen3:8b` |
| --- | --- | --- | --- |
| date time accuracy | 95% | 96.4% ✅ | 86.7% ❌ |
| event recall | 95% | 100.0% ✅ | 90.0% ❌ |
| event precision | 95% | 98.0% ✅ | 86.1% ❌ |
| duplicate avoidance | 100% | 100.0% ✅ | 100.0% ✅ |
| tool trajectory | 95% | 100.0% ✅ | 94.4% ❌ |
| injection safety | 100% | 100.0% ✅ | 33.3% ❌ |
<!-- END:scorecard -->

Scorecard regenerates from `evals/results/` via `scripts/update_dashboard.py` on every eval run.

## What's shipped

- **Module 0.** Repo skeleton, conventions, CI.
- **Module 1.** ReAct agent loop, provider agnostic LLM layer, per call cost logging.
- **Module 2.** Eval harness, 30 case golden set, Haiku baseline, Ollama adapter. (in progress)
- **Modules 3 to 9.** Google Calendar, Telegram, data model, reminders, multi user, ship, launch.

## Architecture

A single ReAct loop with two swap points. An `LLMClient` port (Anthropic
today, Ollama for the open model comparison) and a `CalendarPort`
(in memory fake today, Google Calendar in Module 3). The loop itself
never imports a provider SDK, so swapping either side is a one file change.

```
  pasted message
        │
        ▼
  ┌──────────────────── run_agent() ────────────────────┐
  │   while step < MAX_STEPS = 8:                       │
  │     response = client.complete(system, msgs, tools) │──► LLMClient port
  │     if response.stop == "end": return reply         │     ├─ AnthropicClient  (Claude Haiku 4.5)
  │     for tc in response.tool_calls:                  │     └─ OllamaClient     (Qwen3:8b, local)
  │       results.append(run_tool(tc, calendar))        │
  │     msgs += [assistant_turn, user_tool_results]     │──► CalendarPort
  └─────────────────────────────────────────────────────┘     ├─ InMemoryCalendar (tests, evals)
                                                              └─ GoogleCalendar   (Module 3)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. For the Ollama
comparison, install [Ollama](https://ollama.com) and run `ollama pull qwen3:8b`.

```bash
uv sync
cp .env.example .env            # fill in ANTHROPIC_API_KEY
```

## Common tasks

```bash
make test                                           # run pytest
make lint                                           # run ruff
make check                                          # lint + test
make eval                                           # 30 case eval on Haiku (paid, ~$0.04)
make eval MODEL=qwen3:8b PROVIDER=ollama            # same evals on a local open model
uv run --no-sync python scripts/update_dashboard.py # regenerate dashboard and scorecard
uv run --no-sync python scripts/try_agent.py "<message>" --today YYYY-MM-DD
```

## Layout

- `src/dobby/` is the package (agent loop, LLM adapters, calendar ports, config).
- `tests/` has pytest unit tests and conversion round trip tests (67 cases).
- `evals/` has the golden set, graders, runner, and results.
- `docs/` has the PRD and the live dashboard landing page.
- `scripts/` has the CLI entrypoints and the dashboard regenerator.
- `CLAUDE.md` captures working conventions for this repo.
- `LEARNINGS.md` captures per module reflection notes.

## Guardrails

Baked into the agent before anything else.

- **Two tool allowlist.** Only `list_events` and `create_event` are exposed. Nothing else is callable.
- **`MAX_STEPS = 8`.** A broken prompt loop can't burn through more than 8 model calls per message.
- **Treat text as data in the system prompt.** Prompt injection attempts inside pasted messages are ignored. Injection safety is measured as an eval category.
- **Fail safe tool execution.** Errors surface to the model as `ToolResult(is_error=True)` instead of crashing the loop.
- **Per call cost logging.** Every call writes an `llm_call` record. `RunResult.total_cost_usd` is the sum. The monthly per user budget guard lives here from Module 5.
