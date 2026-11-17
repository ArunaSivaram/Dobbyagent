"""Try the agent from the command line, against an empty fake calendar.

    uv run python scripts/try_agent.py "<message>" --today 2026-10-29

This DOES call the real Claude API and spends a few tenths of a cent.
"""
from __future__ import annotations

import argparse
from datetime import date

from dobby.agent import run_agent
from dobby.calendar.fake import InMemoryCalendar
from dobby.llm import get_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dobby against one message.")
    parser.add_argument("message", help="The pasted message to process.")
    parser.add_argument(
        "--today",
        required=True,
        help="Fixed 'today' in ISO format, e.g. 2026-10-29.",
    )
    args = parser.parse_args()

    today = date.fromisoformat(args.today)
    calendar = InMemoryCalendar()
    client = get_client()

    result = run_agent(args.message, calendar, today, client)

    print("\n--- Reply ---")
    print(result.reply)

    print("\n--- Created events ---")
    if not result.created_events:
        print("(none)")
    for e in result.created_events:
        extras = []
        if e.start_time:
            extras.append(f"{e.start_time.isoformat(timespec='minutes')}")
        if e.end_time:
            extras.append(f"-{e.end_time.isoformat(timespec='minutes')}")
        if e.end_date:
            extras.append(f"through {e.end_date.isoformat()}")
        if e.location:
            extras.append(f"@ {e.location}")
        suffix = f" ({' '.join(extras)})" if extras else ""
        print(f"- {e.date.isoformat()}: {e.title}{suffix}")

    print("\n--- Usage ---")
    print(f"{'step':>4}  {'in':>6}  {'out':>6}  {'cr':>5}  {'cw':>5}  {'cost_usd':>10}  {'ms':>6}")
    for i, u in enumerate(result.usages, 1):
        print(
            f"{i:>4}  {u.input_tokens:>6}  {u.output_tokens:>6}  "
            f"{u.cache_read_tokens:>5}  {u.cache_write_tokens:>5}  "
            f"{u.cost_usd:>10.6f}  {u.latency_ms:>6}"
        )
    print(f"\nTotal: {len(result.usages)} calls, ${result.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
