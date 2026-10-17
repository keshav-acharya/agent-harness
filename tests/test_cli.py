from pathlib import Path

import pytest

from harness.cli import build_parser, check_workdir, format_summary
from harness.loop import RunResult
from harness.types import Usage


def test_workdir_with_env_file_is_refused(tmp_path):
    (tmp_path / ".env").write_text("X=1")
    with pytest.raises(SystemExit):
        check_workdir(tmp_path, force=False)


def test_workdir_with_env_file_allowed_with_force(tmp_path):
    (tmp_path / ".env").write_text("X=1")
    check_workdir(tmp_path, force=True)  # must not raise


def test_missing_workdir_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        check_workdir(tmp_path / "does-not-exist", force=False)


def test_parser_defaults():
    a = build_parser().parse_args(["fix it"])
    assert a.task == "fix it" and a.provider == "anthropic" and a.max_turns == 40


def test_summary_reports_unknown_cost():
    r = RunResult("completed", "done", 3, Usage(100, 50), None, wall_s=2.5)
    s = format_summary(r, Path("t.jsonl"))
    assert "unknown" in s and "completed" in s and "3 turns" in s