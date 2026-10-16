from __future__ import annotations

import json
import time
from pathlib import Path


class Tracer:
    """Append-only JSONL log. Pass None as the path for a no-op tracer (tests)."""

    def __init__(self, path: Path | None, run_id: str = ""):
        self.path = path
        self.run_id = run_id
        self._f = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()

    def emit(self, event: str, **data) -> None:
        if self._f is None:
            return
        record = {"ts": time.time(), "run_id": self.run_id, "event": event, **data}
        self._f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self._f.flush()  # a crash must not lose the trace

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None