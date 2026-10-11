from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from .base import Tool, ToolContext, ToolError

IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                ".mypy_cache", ".pytest_cache", ".ruff_cache"}
MAX_FILE_BYTES = 1_000_000


def _is_binary(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"\0" in f.read(8192)


def _iter_files(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _read_file(args: dict, ctx: ToolContext) -> str:
    path = ctx.resolve(args["path"])
    if not path.exists():
        raise ToolError(f"File not found: {args['path']}")
    if path.is_dir():
        raise ToolError(f"{args['path']} is a directory. Use glob to list files.")
    if _is_binary(path):
        raise ToolError(f"{args['path']} looks like a binary file.")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    offset = max(int(args.get("offset", 1)), 1)
    limit = max(int(args.get("limit", 2000)), 1)
    chunk = lines[offset - 1 : offset - 1 + limit]
    if not chunk:
        return f"[no lines at offset {offset}; file has {len(lines)} lines]"
    body = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start=offset))
    last = offset + len(chunk) - 1
    if last < len(lines):
        body += (f"\n[showing lines {offset}-{last} of {len(lines)}; "
                 f"use offset={last + 1} to continue]")
    return body


def _write_file(args: dict, ctx: ToolContext) -> str:
    path = ctx.resolve(args["path"])
    if path.is_dir():
        raise ToolError(f"{args['path']} is a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = args["content"]
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content.encode('utf-8'))} bytes to {args['path']}"


def _edit_file(args: dict, ctx: ToolContext) -> str:
    path = ctx.resolve(args["path"])
    if not path.is_file():
        raise ToolError(f"File not found: {args['path']}")
    old, new = args["old_string"], args["new_string"]
    if old == "":
        raise ToolError("old_string must not be empty. Use write_file to create files.")
    if old == new:
        raise ToolError("old_string and new_string are identical.")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        raise ToolError(
            "old_string was not found. It must match the file exactly (whitespace included) "
            "and must NOT include read_file's line-number prefixes. Re-read the file and retry."
        )
    if count > 1 and not args.get("replace_all", False):
        raise ToolError(
            f"old_string appears {count} times. Include more surrounding context to make it "
            "unique, or set replace_all=true."
        )
    path.write_text(text.replace(old, new), encoding="utf-8")
    return f"Edited {args['path']} ({count} replacement{'s' if count > 1 else ''})."


def _glob(args: dict, ctx: ToolContext) -> str:
    base = ctx.resolve(args.get("path", "."))
    matches = []
    for p in sorted(base.glob(args["pattern"])):
        rel = p.relative_to(ctx.root)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        matches.append(rel.as_posix() + ("/" if p.is_dir() else ""))
    if not matches:
        return "[no matches]"
    out = "\n".join(matches[:200])
    if len(matches) > 200:
        out += f"\n[{len(matches) - 200} more not shown; narrow the pattern]"
    return out


def _grep(args: dict, ctx: ToolContext) -> str:
    try:
        rx = re.compile(args["pattern"])
    except re.error as e:
        raise ToolError(f"Invalid regex: {e}") from e
    base = ctx.resolve(args.get("path", "."))
    include = args.get("include")
    files = [base] if base.is_file() else _iter_files(base)
    hits: list[str] = []
    for f in files:
        if include and not fnmatch.fnmatch(f.name, include):
            continue
        try:
            if f.stat().st_size > MAX_FILE_BYTES or _is_binary(f):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append(f"{f.relative_to(ctx.root).as_posix()}:{n}:{line[:300]}")
                if len(hits) >= 100:
                    return "\n".join(hits) + "\n[stopped at 100 matches; narrow your search]"
    return "\n".join(hits) if hits else "[no matches]"


READ_FILE = Tool(
    name="read_file",
    description="Read a text file. Returns lines prefixed with line numbers (the prefix is not "
                "part of the file). Use offset/limit for large files.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based first line (default 1)"},
            "limit": {"type": "integer", "description": "Max lines (default 2000)"},
        },
        "required": ["path"],
    },
    run=_read_file,
    read_only=True,
)

WRITE_FILE = Tool(
    name="write_file",
    description="Create a file or completely overwrite it. Prefer edit_file for existing files.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    run=_write_file,
)

EDIT_FILE = Tool(
    name="edit_file",
    description="Replace an exact string in a file. old_string must match exactly once (unless "
                "replace_all is true) and must not include line-number prefixes.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old_string", "new_string"],
    },
    run=_edit_file,
)

GLOB = Tool(
    name="glob",
    description="Find files/directories by glob pattern, e.g. '**/*.py' or 'src/*'.",
    parameters={
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"],
    },
    run=_glob,
    read_only=True,
)

GREP = Tool(
    name="grep",
    description="Search file contents with a Python regex. Returns path:line:text. "
                "'include' filters by filename glob, e.g. '*.py'.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "File or directory (default '.')"},
            "include": {"type": "string"},
        },
        "required": ["pattern"],
    },
    run=_grep,
    read_only=True,
)