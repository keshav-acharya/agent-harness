"""Run the same tasks on several providers and check each result with pytest.

Usage:  python scripts/compare.py anthropic groq --tasks fix_bug implement --trials 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from harness.loop import Agent, Limits
from harness.providers import PROVIDER_NAMES, make_provider
from harness.tools import ToolRegistry, default_tools
from harness.trace import Tracer

TASKS: dict[str, tuple[dict[str, str], str]] = {
    "fix_bug": (
        {
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": (
                "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
            ),
        },
        "The test in this directory fails. Find and fix the bug, then run the test.",
    ),
    "implement": (
        {
            "text.py": '''def slugify(s: str) -> str:
    """Lowercase s, replace each run of non-alphanumeric characters with a single
    hyphen, and strip leading/trailing hyphens.
    Example: slugify("Hello, World!") == "hello-world"
    """
    raise NotImplementedError
''',
            "test_text.py": '''from text import slugify


def test_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_spaces_and_edges():
    assert slugify("  a   b ") == "a-b"


def test_empty():
    assert slugify("") == ""
''',
        },
        "Implement slugify in text.py so the tests in test_text.py pass. Run them to confirm.",
    ),
    "rename": (
        {
            "db.py": 'def get_user_name():\n    return "alice"\n',
            "app.py": (
                "from db import get_user_name\n\n\ndef run():\n"
                "    return get_user_name().upper()\n"
            ),
            "cli.py": (
                "from db import get_user_name\n\n"
                'if __name__ == "__main__":\n    print(get_user_name())\n'
            ),
            "test_rename.py": '''from pathlib import Path

from app import run
from db import fetch_username

OLD = "get_user" + "_name"


def test_new_name_works():
    assert fetch_username() == "alice"
    assert run() == "ALICE"


def test_old_name_is_gone():
    for name in ("db.py", "app.py", "cli.py"):
        assert OLD not in (Path(__file__).parent / name).read_text()
''',
        },
        (
            "Rename the function get_user_name to fetch_username everywhere it is used in this "
                    "project, then run the tests."
        ),
    ),
}


def run_one(provider_name: str, task_name: str, max_turns: int) -> dict:
    files, prompt = TASKS[task_name]
    work = Path(tempfile.mkdtemp(prefix=f"cmp-{task_name}-"))
    for name, content in files.items():
        (work / name).write_text(content, encoding="utf-8")

    provider = make_provider(provider_name)
    trace = Path("traces") / f"cmp-{provider_name}-{task_name}-{work.name}.jsonl"
    tracer = Tracer(trace, work.name)
    agent = Agent(
        provider, ToolRegistry(default_tools()), work, Limits(max_turns=max_turns), tracer
    )
    result = agent.run(prompt)
    tracer.close()

    # The independent check: did the tests REALLY pass?
    check = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=work, capture_output=True, text=True, check=False,
    )
    u = result.usage
    return {
        "provider": provider_name, "model": provider.model, "task": task_name,
        "passed": check.returncode == 0, "status": result.status, "turns": result.turns,
        "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
        "cache_read_tokens": u.cache_read_tokens, "cost_usd": result.cost_usd,
        "wall_s": round(result.wall_s, 1), "error": result.error, "trace": str(trace),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("providers", nargs="+", choices=PROVIDER_NAMES)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=25)
    args = ap.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv()

    rows: list[dict] = []
    for provider_name in args.providers:
        for task_name in args.tasks:
            for trial in range(1, args.trials + 1):
                print(f"running {provider_name} / {task_name} / trial {trial} ...", flush=True)
                try:
                    rows.append(run_one(provider_name, task_name, args.max_turns))
                except Exception as e:  # noqa: BLE001 - one broken run must not stop the batch
                    print(f"  CRASHED: {type(e).__name__}: {e}")
                    rows.append({
                        "provider": provider_name, "task": task_name, "passed": False,
                        "status": "crash", "turns": 0, "input_tokens": 0, "output_tokens": 0,
                        "cost_usd": None, "wall_s": 0.0, "error": str(e),
                    })

    print(f"\n{'provider':<10} {'task':<10} {'pass':<5} {'status':<10} {'turns':>5} "
          f"{'in':>8} {'out':>6} {'cost':>9} {'secs':>6}")
    for r in rows:
        cost = "?" if r["cost_usd"] is None else f"${r['cost_usd']:.4f}"
        print(f"{r['provider']:<10} {r['task']:<10} {'YES' if r['passed'] else 'no':<5} "
              f"{r['status']:<10} {r['turns']:>5} {r['input_tokens']:>8} "
              f"{r['output_tokens']:>6} {cost:>9} {r['wall_s']:>6}")

    print()
    for name in args.providers:
        mine = [r for r in rows if r["provider"] == name]
        print(f"{name}: {sum(r['passed'] for r in mine)}/{len(mine)} passed")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"compare-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()