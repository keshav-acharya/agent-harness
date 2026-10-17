from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def summarize(path: Path) -> str:
    events = load_events(path)
    out: list[str] = []

    start = next((e for e in events if e["event"] == "run_start"), None)
    if start:
        out.append(f"task:  {start['task'][:100]}")
        out.append(f"model: {start['provider']}/{start['model']}")

    results_by_turn: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e["event"] == "tool_result":
            results_by_turn[e["turn"]].append(e)

    out.append("")
    out.append(f"{'turn':>4} {'in':>8} {'out':>6} {'cached':>8} {'secs':>6}  tool calls")
    for e in events:
        if e["event"] != "model_response":
            continue
        u = e["usage"]
        results = results_by_turn.get(e["turn"], [])
        names = []
        for i, call in enumerate(e["tool_calls"]):
            failed = i < len(results) and results[i]["is_error"]
            names.append(call["name"] + (" (ERR)" if failed else ""))
        out.append(
            f"{e['turn']:>4} {u['input_tokens']:>8} {u['output_tokens']:>6} "
            f"{u['cache_read_tokens']:>8} {e['latency_s']:>6.1f}  "
            f"{', '.join(names) or '(final answer)'}"
        )

    all_results = [e for e in events if e["event"] == "tool_result"]
    if all_results:
        big = max(all_results, key=lambda e: len(e["content"]))
        out.append(f"\nlargest tool result: {big['name']}, {len(big['content'])} chars (turn {big['turn']})")

    retries = sum(1 for e in events if e["event"] == "provider_retry")
    if retries:
        out.append(f"provider retries: {retries}")

    end = next((e for e in reversed(events) if e["event"] == "run_end"), None)
    if end:
        cost = "unknown" if end["cost_usd"] is None else f"${end['cost_usd']:.4f}"
        out.append(
            f"\nstatus={end['status']} turns={end['turns']} cost={cost} wall={end['wall_s']:.1f}s"
        )
        if end.get("error"):
            out.append(f"error: {end['error']}")
    else:
        out.append("\n(no run_end event: the run was interrupted or crashed)")
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        files = sorted(Path("traces").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            sys.exit("No traces found in ./traces")
        path = files[-1]  # the newest one
    print(summarize(path))