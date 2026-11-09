"""Print one turn of a trace in full.

Usage:  python scripts/show_turn.py traces/<file>.jsonl 3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/show_turn.py <trace.jsonl> <turn-number>")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path, turn = Path(sys.argv[1]), int(sys.argv[2])
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("turn") != turn:
            continue
        if event["event"] == "model_response":
            print(f"=== MODEL, turn {turn} (stop reason: {event['stop_reason']})")
            if event["text"].strip():
                print("SAID:", event["text"].strip())
            for call in event["tool_calls"]:
                print(f"ASKED FOR: {call['name']}")
                print(json.dumps(call["input"], indent=2))
                if call.get("parse_error"):
                    print("BAD JSON:", call["parse_error"])
        elif event["event"] == "tool_result":
            state = "ERROR" if event["is_error"] else "ok"
            print(f"\n=== RESULT of {event['name']} [{state}]")
            print(event["content"][:2500])


if __name__ == "__main__":
    main()