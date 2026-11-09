from __future__ import annotations

import os
import signal
import subprocess
import sys

from .base import Tool, ToolContext, ToolError, truncate

IS_WINDOWS = sys.platform == "win32"


def _clean_env() -> dict[str, str]:
    """Child processes inherit our environment, which load_dotenv() filled with API keys.

    Return a copy without any variable whose name contains API_KEY.
    """
    return {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process AND everything it started."""
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        os.killpg(proc.pid, signal.SIGKILL)


def _bash(args: dict, ctx: ToolContext) -> str:
    cmd = args["command"]
    if IS_WINDOWS and ("\n" in cmd or "\r" in cmd):
        raise ToolError(
            "This command contains a line break, which the Windows shell (cmd.exe) does not "
            "handle: it ignores the rest of the command and may silently do nothing. Put "
            "multi-line code in a file with write_file and run that file, or keep the whole "
            "command on one line (use && between steps)."
        )
    timeout = min(max(int(args.get("timeout", 120)), 1), 600)

    kwargs: dict = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=ctx.root,
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.communicate()
        raise ToolError(f"Command timed out after {timeout}s and was killed.") from None

    combined = out + (f"\n[stderr]\n{err}" if err else "")
    if not combined.strip():
        combined = "[no output]"  # silence must not look like success
    return f"{truncate(combined)}\n[exit code {proc.returncode}]"


BASH = Tool(
    name="bash",
    description=(
        "Run a shell command in the project directory. Returns combined output and the "
        "exit code. Each call starts a fresh shell: cd and environment changes do NOT "
        "persist between calls, so chain steps with && in one command. "
        "Commands cannot read stdin. Default timeout 120s (max 600)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Seconds (default 120)"},
        },
        "required": ["command"],
    },
    run=_bash,
)