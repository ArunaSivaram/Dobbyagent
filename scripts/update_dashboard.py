"""Regenerate docs/dashboard.html and the README scorecard from the latest
per-model JSON in evals/results/.

Usage: `uv run --no-sync python scripts/update_dashboard.py`

Idempotent. Safe to run any time; picks up whichever model JSONs exist.
"""
from __future__ import annotations

import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path("evals/results")
DASHBOARD_PATH = Path("docs/index.html")
README_PATH = Path("README.md")

THRESHOLDS: dict[str, float] = {
    "date_time_accuracy": 0.95,
    "event_recall": 0.95,
    "event_precision": 0.95,
    "duplicate_avoidance": 1.00,
    "tool_trajectory": 0.95,
    "injection_safety": 1.00,
}

MODULES: list[dict[str, object]] = [
    {"id": 0, "name": "Setup",            "status": "done"},
    {"id": 1, "name": "Agent core",       "status": "done"},
    {"id": 2, "name": "Evals + Ollama",   "status": "in_progress"},
    {"id": 3, "name": "Google Calendar",  "status": "todo"},
    {"id": 4, "name": "Telegram + files", "status": "todo"},
    {"id": 5, "name": "Data + cost",      "status": "todo"},
    {"id": 6, "name": "Reminders",        "status": "todo"},
    {"id": 7, "name": "Multi-user",       "status": "todo"},
    {"id": 8, "name": "Ship",             "status": "todo"},
    {"id": 9, "name": "Launch",           "status": "todo"},
]


def main() -> None:
    models = _summarise_latest_per_model()
    data = _build_data_block(models)
    _patch_dashboard(data)
    _patch_readme(data)
    print(f"Updated {DASHBOARD_PATH} and {README_PATH}")
    print(f"Models: {', '.join(m['id'] for m in models) or '(none found)'}")


# --- IO -------------------------------------------------------------------

def _summarise_latest_per_model() -> list[dict]:
    """Group result JSONs by model; keep the newest per model (by mtime)."""
    if not RESULTS_DIR.exists():
        return []
    latest: dict[str, tuple[float, dict]] = {}
    for path in RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        model = data.get("model")
        if not model:
            continue
        mtime = path.stat().st_mtime
        if model not in latest or mtime > latest[model][0]:
            latest[model] = (mtime, data)
    return [_summarise_one(d) for _, d in latest.values()]


def _summarise_one(model_data: dict) -> dict:
    all_runs = [r for cr in model_data.get("results", []) for r in cr.get("runs", [])]
    costs = [r["cost_usd"] for r in all_runs]
    latencies = [r["latency_ms"] for r in all_runs if r.get("latency_ms", 0) > 0]
    scores = {
        metric: (info["mean"] if info.get("n", 0) > 0 else None)
        for metric, info in model_data.get("aggregate", {}).items()
    }
    return {
        "id": model_data["model"],
        "provider": model_data["provider"],
        "status": "measured",
        "runs_per_case": model_data.get("runs_per_case"),
        "case_count": model_data.get("case_count"),
        "total_cost_usd": round(sum(costs), 4),
        "cost_per_message_usd": statistics.mean(costs) if costs else 0.0,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "scores": scores,
        "thresholds_passed": sum(
            1
            for metric, value in scores.items()
            if value is not None and value >= THRESHOLDS[metric]
        ),
        "by_category": {
            cat: {
                "mean_latency_ms": row.get("mean_latency_ms"),
                "mean_cost_usd": row.get("mean_cost_usd"),
            }
            for cat, row in model_data.get("by_category", {}).items()
        },
    }


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _build_data_block(models: list[dict]) -> dict:
    return {
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "thresholds": THRESHOLDS,
        "models": models,
        "modules": MODULES,
    }


# --- dashboard.html -------------------------------------------------------

_DASHBOARD_PATTERN = re.compile(
    r'(<script id="eval-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)


def _patch_dashboard(data: dict) -> None:
    html = DASHBOARD_PATH.read_text()
    pretty = json.dumps(data, indent=2)
    replacement = f'\\1\n{pretty}\n  \\3'
    new = _DASHBOARD_PATTERN.sub(replacement, html, count=1)
    if new == html:
        raise SystemExit(
            f"Could not find data block in {DASHBOARD_PATH}. "
            'Expected a <script id="eval-data" type="application/json"> tag.'
        )
    DASHBOARD_PATH.write_text(new)


# --- README.md ------------------------------------------------------------

_README_MARKERS = ("<!-- BEGIN:scorecard -->", "<!-- END:scorecard -->")


def _patch_readme(data: dict) -> None:
    if not README_PATH.exists():
        return
    readme = README_PATH.read_text()
    start, end = _README_MARKERS
    if start not in readme or end not in readme:
        print(f"(skipping README — markers {start!r} / {end!r} not found)")
        return
    block = _render_readme_block(data)
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end), re.DOTALL
    )
    new = pattern.sub(start + "\n" + block + "\n" + end, readme, count=1)
    README_PATH.write_text(new)


def _render_readme_block(data: dict) -> str:
    models = data["models"]
    lines: list[str] = []
    lines.append(f"_Latest evals · updated {data['updated_at']}_")
    lines.append("")
    lines.append("| Model | Hosting | Cost / msg | Latency p95 | Thresholds passed | Status |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if not models:
        lines.append("| _(no results yet — run `make eval`)_ | | | | | |")
    else:
        for m in models:
            cost = f"${m['cost_per_message_usd']:.6f}" if m["cost_per_message_usd"] is not None else "—"
            p95 = f"{int(m['latency_ms_p95']):,} ms" if m["latency_ms_p95"] else "—"
            passed = f"{m['thresholds_passed']} / {len(THRESHOLDS)}"
            lines.append(f"| `{m['id']}` | {m['provider']} | {cost} | {p95} | {passed} | {m['status']} |")
    lines.append("")
    lines.append("**Per-metric scores:**")
    lines.append("")
    header = ["Metric", "Threshold"] + [f"`{m['id']}`" for m in models]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for metric, threshold in THRESHOLDS.items():
        row = [metric.replace("_", " "), f"{threshold * 100:.0f}%"]
        for m in models:
            v = m["scores"].get(metric)
            if v is None:
                row.append("—")
            else:
                mark = "✅" if v >= threshold else "❌"
                row.append(f"{v * 100:.1f}% {mark}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
