from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .loop import Agent, Limits, RunResult
from .pricing import PRICES
from .providers import PROVIDER_NAMES, make_provider
from .tools import ToolRegistry, default_tools
from .trace import Tracer


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="harness", description="Minimal coding-agent harness")
    ap.add_argument("task", help="what the agent should do (put it in quotes)")
    ap.add_argument("--provider", default="anthropic", choices=PROVIDER_NAMES)
    ap.add_argument("--model", help="defaults to <PROVIDER>_MODEL from .env")
    ap.add_argument("--workdir", default=".", help="folder the agent works in")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--max-cost", type=float, default=2.0, help="USD; needs a price in pricing.py")
    ap.add_argument("--max-seconds", type=float, default=900)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--trace-dir", default="traces")
    ap.add_argument("--quiet", action="store_true", help="only print the final summary")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow a workdir that contains .env")
    return ap


def check_workdir(workdir: Path, force: bool) -> None:
    if not workdir.is_dir():
        sys.exit(f"error: --workdir '{workdir}' is not a directory")
    if (workdir / ".env").exists() and not force:
        sys.exit(
            "error: the working directory contains a .env file, so the agent could read your "
            "API keys. Use a scratch folder, or pass --force if you really mean it."
        )


def make_printer(color: bool):
    def paint(code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    def on_event(event: str, d: dict) -> None:
        if event == "model_response":
            text = d["text"].strip()
            if text:
                print("\n" + paint("1", text))
            for c in d["tool_calls"]:
                args = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in c["input"].items())
                line = f"-> {c['name']}({args})"
                if c.get("parse_error"):
                    line += f"  [bad JSON: {c['parse_error']}]"
                print(paint("36", line))
        elif event == "tool_result":
            first = d["content"].strip().splitlines()[:1]
            mark = paint("31", "ERR") if d["is_error"] else paint("32", "ok ")
            print(f"   {mark} {(first[0] if first else '')[:100]}")
        elif event == "provider_retry":
            print(paint("33", f"   model error, retrying (attempt {d['attempt']})"))

    return on_event


def format_summary(result: RunResult, trace_path: Path) -> str:
    u = result.usage
    cost = (
        f"${result.cost_usd:.4f}"
        if result.cost_usd is not None
        else "unknown (add a price in pricing.py)"
    )
    lines = [
        "",
        f"[{result.status}] {result.turns} turns in {result.wall_s:.1f}s",
        (
            f"tokens: {u.input_tokens} in, {u.output_tokens} out, "
            f"{u.cache_read_tokens} cache-read, {u.cache_write_tokens} cache-write"
        ),   
        f"cost:   {cost}",
        f"trace:  {trace_path}",
    ]
    if result.error:
        lines.append(f"error:  {result.error}")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()

    load_dotenv(find_dotenv(usecwd=True))  # a .env in the folder you run from
    load_dotenv()                          # fallback: the project's own .env
    for stream in (sys.stdout, sys.stderr):  # Windows consoles choke on some characters
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    workdir = Path(args.workdir).resolve()
    check_workdir(workdir, args.force)

    try:
        provider = make_provider(args.provider, args.model)
    except Exception as e:  # noqa: BLE001 - bad model name, missing key, etc: show it cleanly
        sys.exit(f"error: {e}")

    if provider.model not in PRICES:
        print(
            f"warning: no price for '{provider.model}' in pricing.py. Cost will be unknown "
            "and --max-cost is NOT enforced.",
            file=sys.stderr,
        )

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{provider.name}"
    trace_path = Path(args.trace_dir) / f"{run_id}.jsonl"
    tracer = Tracer(trace_path, run_id)

    color = not args.no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    agent = Agent(
        provider,
        ToolRegistry(default_tools()),
        workdir,
        Limits(
            max_turns=args.max_turns,
            max_cost_usd=args.max_cost,
            max_wall_s=args.max_seconds,
            max_output_tokens=args.max_output_tokens,
        ),
        tracer,
        None if args.quiet else make_printer(color),
    )

    try:
        result = agent.run(args.task)
    except KeyboardInterrupt:
        tracer.emit("interrupted")
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
    finally:
        tracer.close()

    print(format_summary(result, trace_path))
    sys.exit(0 if result.status == "completed" else 1)