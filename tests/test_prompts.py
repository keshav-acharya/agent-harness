from pathlib import Path

from harness.prompts import build_system_prompt


def test_windows_prompt_warns_about_single_line_commands():
    prompt = build_system_prompt(Path("."), os_name="win32")
    assert "cmd.exe" in prompt and "write_file" in prompt


def test_other_platforms_use_sh():
    assert "runs commands with sh" in build_system_prompt(Path("."), os_name="linux")