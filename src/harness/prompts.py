from __future__ import annotations

import sys
from pathlib import Path

_TEMPLATE = """You are a coding agent working inside a project directory.
Working directory: {root}
Operating system: {platform}. The bash tool runs commands with {shell}.

Rules:
- Explore before changing anything: use glob, grep and read_file to understand the code.
- Use edit_file for changes to existing files. Read the file first so old_string matches exactly.
- After making changes, verify them by running the relevant tests or commands with bash.
- If a tool returns an error, read it carefully and change your approach. Never repeat an identical failing call.
- Keep going until the task is fully done, then reply with a short summary of what you changed.
- Do not ask the user questions. Make reasonable assumptions and state them in your summary.
"""


def build_system_prompt(root: Path) -> str:
    if sys.platform == "win32":
        platform = "Windows"
        shell = "cmd.exe (use 'dir' and 'type', not 'ls' and 'cat'; run Python as 'python')"
    else:
        platform = sys.platform
        shell = "sh"
    return _TEMPLATE.format(root=root, platform=platform, shell=shell)