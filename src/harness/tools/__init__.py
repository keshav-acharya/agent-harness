from __future__ import annotations

import jsonschema

from ..types import ToolCall, ToolResult
from .base import Tool, ToolContext, ToolError, truncate
from .fs import EDIT_FILE, GLOB, GREP, READ_FILE, WRITE_FILE
from .shell import BASH

MAX_RESULT_CHARS = 20_000


def default_tools() -> list[Tool]:
    return [READ_FILE, WRITE_FILE, EDIT_FILE, GLOB, GREP, BASH]


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        def err(msg: str) -> ToolResult:
            return ToolResult(call.id, msg, is_error=True)

        tool = self._tools.get(call.name)
        if tool is None:
            return err(f"Unknown tool '{call.name}'. Available: {', '.join(self._tools)}")
        if call.parse_error:
            return err(f"Your arguments were not valid JSON ({call.parse_error}). Retry the call.")
        try:
            jsonschema.validate(call.input, tool.parameters)
        except jsonschema.ValidationError as e:
            return err(f"Invalid arguments for {call.name}: {e.message}")
        try:
            return ToolResult(call.id, truncate(tool.run(call.input, ctx), MAX_RESULT_CHARS))
        except ToolError as e:
            return err(str(e))
        except Exception as e:  # noqa: BLE001 - a tool bug must never crash the loop
            return err(f"Internal error in {call.name}: {type(e).__name__}: {e}")