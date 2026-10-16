from harness import pricing
from harness.pricing import Price, cost_usd
from harness.types import Message, ToolCall, Usage


def test_usage_add_accumulates():
    a = Usage(input_tokens=10, output_tokens=5)
    a.add(Usage(input_tokens=1, output_tokens=2, cache_read_tokens=3))
    assert (a.input_tokens, a.output_tokens, a.cache_read_tokens) == (11, 7, 3)


def test_messages_do_not_share_default_lists():
    m1, m2 = Message("assistant"), Message("assistant")
    m1.tool_calls.append(ToolCall("1", "x", {}))
    assert m2.tool_calls == []


def test_cost_unknown_model_is_none():
    assert cost_usd("no-such-model", Usage(100, 100)) is None


def test_cost_math(monkeypatch):
    monkeypatch.setitem(pricing.PRICES, "m", Price(input=2.0, output=10.0))
    # 1M input tokens * $2 + 1M output tokens * $10 = $12
    assert cost_usd("m", Usage(1_000_000, 1_000_000)) == 12.0