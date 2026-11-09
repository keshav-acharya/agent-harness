"""Print the newest results/compare-*.json as a table.

Usage:  python scripts/show_results.py [path]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        files = sorted(Path("results").glob("compare-*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            sys.exit("no results/compare-*.json found")
        path = files[-1]
    rows = json.loads(path.read_text(encoding="utf-8"))
    print(path)
    print(f"{'task':<10} {'passed':<7} {'status':<10} {'turns':>5} "
          f"{'in':>8} {'secs':>6} {'waits':>6}")
    for r in rows:
        print(
            f"{r['task']:<10} {r['passed']!s:<7} {r['status']:<10} {r['turns']:>5} "
            f"{r['input_tokens']:>8} {r['wall_s']:>6} {r.get('retry_wait_s', 0):>6}"
        )
    odd = [r for r in rows if r["status"] == "completed" and not r["passed"]]
    if odd:
        print(f"\n{len(odd)} run(s) said completed but FAILED the independent check:")
        for r in odd:
            print(f"  {r['task']}: {r.get('trace', '')}")


if __name__ == "__main__":
    main()