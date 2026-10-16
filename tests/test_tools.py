import pytest

from harness.tools import ToolRegistry, default_tools
from harness.tools.base import ToolContext
from harness.types import ToolCall


@pytest.fixture
def env(tmp_path):
    reg, ctx = ToolRegistry(default_tools()), ToolContext(tmp_path)

    def call(name, **kw):
        return reg.execute(ToolCall("id1", name, kw), ctx)

    return call, tmp_path


def test_write_then_read_has_line_numbers(env):
    call, _ = env
    call("write_file", path="a.txt", content="one\ntwo\n")
    r = call("read_file", path="a.txt")
    assert not r.is_error and "     2\ttwo" in r.content


def test_edit_unique_ok(env):
    call, root = env
    (root / "f.py").write_text("x = 1\ny = 2\n")
    assert not call("edit_file", path="f.py", old_string="x = 1", new_string="x = 9").is_error
    assert (root / "f.py").read_text() == "x = 9\ny = 2\n"


def test_edit_ambiguous_fails_and_leaves_file(env):
    call, root = env
    (root / "f.py").write_text("a\na\n")
    r = call("edit_file", path="f.py", old_string="a", new_string="b")
    assert r.is_error and "2 times" in r.content
    assert (root / "f.py").read_text() == "a\na\n"


def test_edit_replace_all(env):
    call, root = env
    (root / "f.py").write_text("a\na\n")
    call("edit_file", path="f.py", old_string="a", new_string="b", replace_all=True)
    assert (root / "f.py").read_text() == "b\nb\n"


def test_path_escape_blocked(env):
    call, _ = env
    assert call("read_file", path="../../etc/passwd").is_error
    assert call("write_file", path="/tmp/evil.txt", content="x").is_error


def test_grep_and_glob(env):
    call, root = env
    (root / "pkg").mkdir()
    (root / "pkg" / "m.py").write_text("def foo():\n    pass\n")
    assert "pkg/m.py:1:def foo" in call("grep", pattern=r"def \w+").content
    assert "pkg/m.py" in call("glob", pattern="**/*.py").content


def test_bash_exit_code(env):
    call, _ = env
    r = call("bash", command='python -c "import sys; print(\'hi\'); sys.exit(3)"')
    assert "hi" in r.content and "[exit code 3]" in r.content


def test_bash_timeout(env):
    call, _ = env
    r = call("bash", command='python -c "import time; time.sleep(30)"', timeout=1)
    assert r.is_error and "timed out" in r.content


def test_invalid_args_and_unknown_tool(env):
    call, _ = env
    assert call("read_file").is_error      # missing required 'path'
    assert call("nope").is_error


def test_bash_does_not_leak_api_keys(env, monkeypatch):
    call, _ = env
    monkeypatch.setenv("FAKE_API_KEY", "super-secret")
    r = call("bash", command='python -c "import os; print(os.environ.get(\'FAKE_API_KEY\'))"')
    assert "super-secret" not in r.content and "None" in r.content