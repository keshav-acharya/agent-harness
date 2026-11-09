"""Create (or reset) a tiny buggy project next to the repo.

Usage:  python scripts/make_demo.py [folder]      (default: ../smoke-demo)
"""
from __future__ import annotations

import sys
from pathlib import Path

FILES = {
    "calc.py": "def add(a, b):\n    return a - b\n",
    "test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
}


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("..") / "smoke-demo"
    target.mkdir(parents=True, exist_ok=True)
    for name, text in FILES.items():
        (target / name).write_bytes(text.encode("utf-8"))  # bytes: exact line endings
    print(f"demo project ready: {target.resolve()}")


if __name__ == "__main__":
    main()