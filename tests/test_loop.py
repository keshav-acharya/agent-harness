import json

from harness import pricing
from harness.loop import Agent, Limits
from harness.pricing import Price
from harness.tools import ToolRegistry, default_tools
from harness.trace import Tracer
from harness.types import ModelResponse, ToolCall, Usage


class FakeProvider:
    """Pretends to be an AI: returns pre-written responses, no network, no cost."""

    name, model = "fake", "fake-model"

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def complete(self, *, system, messages, tools, max_tokens):
        self.calls.append(list(messages))  # snapshot: the loop keeps mutating the list
        return self.responses.pop(0)


def resp(text="", calls=(), stop=None):
    calls = list(calls)
    return ModelResponse(text, calls, Usage(10, 5), stop or ("tool_use" if calls else "end_turn"))


def make(tmp_path, responses, tracer=None, **limits):
    p = FakeProvider(responses)
    agent = Agent(p, ToolRegistry(default_tools()), tmp_path, Limits(**limits), tracer)
    return p, agent


def test_tool_then_finish(tmp_path):
    _, agent = make(tmp_path, [
        resp(calls=[ToolCall("1", "write_file", {"path": "hi.txt", "content": "hello"})]),
        resp("Done."),
    ])
    r = agent.run("make a file")
    assert r.status == "completed" and r.turns == 2 and r.final_text == "Done."
    assert (tmp_path / "hi.txt").read_text() == "hello"
    assert r.usage.input_tokens == 20 and r.usage.output_tokens == 10


def test_unknown_tool_error_is_fed_back(tmp_path):
    p, agent = make(tmp_path, [resp(calls=[ToolCall("1", "nope", {})]), resp("ok")])
    assert agent.run("x").status == "completed"
    assert p.calls[1][-1].tool_results[0].is_error


def test_malformed_args_do_not_crash(tmp_path):
    bad = ToolCall("1", "read_file", {}, parse_error="Expecting value")
    p, agent = make(tmp_path, [resp(calls=[bad]), resp("ok")])
    assert agent.run("x").status == "completed"
    assert "not valid JSON" in p.calls[1][-1].tool_results[0].content


def test_every_parallel_call_gets_a_result_in_order(tmp_path):
    calls = [ToolCall("a", "glob", {"pattern": "*"}), ToolCall("b", "glob", {"pattern": "*.py"})]
    p, agent = make(tmp_path, [resp(calls=calls), resp("ok")])
    agent.run("x")
    results = p.calls[1][-1].tool_results
    assert [r.call_id for r in results] == ["a", "b"]


def test_max_turns_stops_runaway(tmp_path):
    loop = [resp(calls=[ToolCall(str(i), "glob", {"pattern": "*"})]) for i in range(5)]
    _, agent = make(tmp_path, loop, max_turns=3)
    r = agent.run("x")
    assert r.status == "max_turns" and r.turns == 3


def test_max_tokens_is_truncated_not_executed(tmp_path):
    cut = resp(
        "half a thought",
        [ToolCall("1", "write_file", {"path": "x", "content": "y"})],
        stop="max_tokens",
    )
    _, agent = make(tmp_path, [cut])
    r = agent.run("x")
    assert r.status == "truncated"
    assert not (tmp_path / "x").exists()


def test_cost_limit_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setitem(pricing.PRICES, "fake-model", Price(input=1000.0, output=1000.0))
    two = [resp(calls=[ToolCall("1", "glob", {"pattern": "*"})]), resp("never reached")]
    _, agent = make(tmp_path, two, max_cost_usd=0.0001)
    r = agent.run("x")
    assert r.status == "max_cost" and r.turns == 1 and r.cost_usd > 0.0001


def test_provider_exception_becomes_error_status(tmp_path):
    class Boom(FakeProvider):
        def complete(self, **kw):
            raise RuntimeError("network down")

    agent = Agent(Boom([]), ToolRegistry(default_tools()), tmp_path)
    r = agent.run("x")
    assert r.status == "error" and "network down" in r.error and r.turns == 0


def test_trace_file_has_expected_events(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    trace = tmp_path / "t.jsonl"
    tracer = Tracer(trace, "r1")
    _, agent = make(work, [resp("hi")], tracer=tracer)
    agent.run("x")
    tracer.close()
    events = [json.loads(line)["event"] for line in trace.read_text(encoding="utf-8").splitlines()]
    assert events == ["run_start", "model_response", "run_end"]