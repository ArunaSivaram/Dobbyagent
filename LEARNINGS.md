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

## Module 3

## Module 4

## Module 4b

## Module 5

## Module 6

## Module 7

## Module 8

## Module 9
