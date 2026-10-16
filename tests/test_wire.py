from types import SimpleNamespace

from harness.providers.anthropic_provider import AnthropicProvider
from harness.providers.chat_completions import ChatCompletionsProvider
from harness.providers.groq_provider import GroqProvider
from harness.providers.openai_provider import OpenAIProvider
from harness.types import Message, ToolCall, ToolResult

HISTORY = [
    Message("user", text="hi"),
    Message("assistant", text="looking", tool_calls=[ToolCall("c1", "glob", {"pattern": "*"})]),
    Message(
        "user",
        tool_results=[ToolResult("c1", "a.py"), ToolResult("c2", "boom", is_error=True)],
    ),
]


# ---------- Anthropic ----------
def test_anthropic_wire_shapes():
    wire = AnthropicProvider._to_wire(HISTORY)
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[1]["content"][1] == {
        "type": "tool_use", "id": "c1", "name": "glob", "input": {"pattern": "*"}}
    blocks = wire[2]["content"]  # both results live in ONE user message
    assert [b["type"] for b in blocks] == ["tool_result", "tool_result"]
    assert blocks[1]["is_error"] is True


def test_anthropic_tool_results_come_before_text():
    wire = AnthropicProvider._to_wire(
        [Message("user", text="extra", tool_results=[ToolResult("c1", "ok")])])
    assert [b["type"] for b in wire[0]["content"]] == ["tool_result", "text"]


def test_anthropic_never_sends_empty_content():
    wire = AnthropicProvider._to_wire([Message("assistant")])
    assert wire[0]["content"] == [{"type": "text", "text": "(empty)"}]


def test_anthropic_parses_response():
    api_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="let me look"),
            SimpleNamespace(type="tool_use", id="t1", name="glob", input={"pattern": "*"}),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                              cache_read_input_tokens=3, cache_creation_input_tokens=None),
        stop_reason="tool_use",
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: api_response))
    r = AnthropicProvider("m", client=client).complete(
        system="s", messages=[Message("user", text="hi")], tools=[], max_tokens=100)
    assert r.text == "let me look" and r.stop_reason == "tool_use"
    assert r.tool_calls[0].name == "glob" and r.tool_calls[0].input == {"pattern": "*"}
    u = r.usage
    assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens) == (10, 3, 0)


# ---------- OpenAI / Groq ----------
def fake_chat_client(message, finish="tool_calls", prompt=100, completion=20, cached=40):
    api_response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
    )
    completions = SimpleNamespace(create=lambda **kw: api_response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def tool_call(arguments):
    return SimpleNamespace(id="c1", function=SimpleNamespace(name="glob", arguments=arguments))


def test_chat_wire_shapes():
    wire = ChatCompletionsProvider._to_wire("sys", HISTORY)
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool", "tool"]
    assert wire[2]["tool_calls"][0]["function"]["arguments"] == '{"pattern": "*"}'
    assert wire[3]["tool_call_id"] == "c1" and wire[3]["content"] == "a.py"
    assert wire[4]["content"] == "ERROR: boom"  # no is_error flag in this format


def test_openai_normalises_cached_tokens():
    msg = SimpleNamespace(content=None, tool_calls=[tool_call('{"pattern": "*"}')])
    r = OpenAIProvider("m", client=fake_chat_client(msg)).complete(
        system="s", messages=[Message("user", text="hi")], tools=[], max_tokens=100)
    assert r.stop_reason == "tool_use" and r.tool_calls[0].input == {"pattern": "*"}
    u = r.usage
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (60, 40, 20)


def test_malformed_json_arguments_set_parse_error_instead_of_raising():
    msg = SimpleNamespace(content="", tool_calls=[tool_call('{"pattern": ')])
    r = GroqProvider("m", client=fake_chat_client(msg)).complete(
        system="s", messages=[Message("user", text="hi")], tools=[], max_tokens=100)
    assert r.tool_calls[0].parse_error is not None and r.tool_calls[0].input == {}


def test_non_object_arguments_are_rejected():
    msg = SimpleNamespace(content="", tool_calls=[tool_call("[1, 2]")])
    r = OpenAIProvider("m", client=fake_chat_client(msg)).complete(
        system="s", messages=[Message("user", text="hi")], tools=[], max_tokens=100)
    assert "JSON object" in r.tool_calls[0].parse_error


def test_groq_and_openai_share_logic_but_not_names():
    assert GroqProvider("m", client=object()).name == "groq"
    assert OpenAIProvider("m", client=object()).name == "openai"
    assert issubclass(GroqProvider, ChatCompletionsProvider)