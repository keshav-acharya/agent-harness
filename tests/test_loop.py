

import json
from types import SimpleNamespace

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


class Flaky(FakeProvider):
    """Raises a recoverable error `failures` times, then behaves normally."""

    def __init__(self, responses, failures):
        super().__init__(responses)
        self.failures = failures

    def complete(self, **kw):
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("400 tool_use_failed: Failed to call a function")
        return super().complete(**kw)


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

def test_recoverable_provider_error_is_retried(tmp_path):
    events = []
    agent = Agent(
        Flaky([resp("ok")], failures=2), ToolRegistry(default_tools()), tmp_path,
        on_event=lambda e, d: events.append(e),
    )
    r = agent.run("x")
    assert r.status == "completed" and r.turns == 1
    assert events.count("provider_retry") == 2


def test_provider_retries_are_limited(tmp_path):
    agent = Agent(Flaky([resp("ok")], failures=5), ToolRegistry(default_tools()), tmp_path)
    r = agent.run("x")
    assert r.status == "error" and "tool_use_failed" in r.error


def test_unrecoverable_error_is_not_retried(tmp_path):
    class BadKey(FakeProvider):
        n = 0

        def complete(self, **kw):
            BadKey.n += 1
            raise RuntimeError("401 invalid api key")

    r = Agent(BadKey([]), ToolRegistry(default_tools()), tmp_path).run("x")
    assert r.status == "error" and BadKey.n == 1


def test_repeated_identical_failure_gets_a_warning(tmp_path):
    def bad(i):
        return resp(calls=[ToolCall(str(i), "read_file", {"path": "nope.txt"})])

    p, agent = make(tmp_path, [bad(1), bad(2), bad(3), resp("ok")])
    agent.run("x")
    first = p.calls[1][-1].tool_results[0]
    third = p.calls[3][-1].tool_results[0]
    assert "identical" not in first.content
    assert "identical" in third.content

class RateLimitError(Exception):
    """Same class name as the SDKs' error, which is how the loop recognises it."""

    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        if retry_after is not None:
            self.response = SimpleNamespace(headers={"retry-after": str(retry_after)})


class Throttled(FakeProvider):
    def __init__(self, responses, failures, retry_after=None):
        super().__init__(responses)
        self.failures, self.retry_after = failures, retry_after

    def complete(self, **kw):
        if self.failures > 0:
            self.failures -= 1
            raise RateLimitError("429 rate limited", self.retry_after)
        return super().complete(**kw)


def _throttled_run(tmp_path, **kw):
    events, slept = [], []
    agent = Agent(
        Throttled([resp("ok")], **kw), ToolRegistry(default_tools()), tmp_path,
        on_event=lambda e, d: events.append((e, d)),
    )
    agent._sleep = slept.append  # record instead of really sleeping
    return agent.run("x"), events, slept


def test_rate_limit_backs_off_and_is_traced(tmp_path):
    r, events, slept = _throttled_run(tmp_path, failures=2)
    assert r.status == "completed" and slept == [2.0, 4.0]
    assert [e for e, _ in events].count("transient_retry") == 2


def test_server_retry_after_is_respected(tmp_path):
    r, _, slept = _throttled_run(tmp_path, failures=1, retry_after=7)
    assert r.status == "completed" and slept == [7.0]


def test_huge_retry_after_stops_instead_of_waiting(tmp_path):
    r, _, slept = _throttled_run(tmp_path, failures=1, retry_after=3600)
    assert r.status == "error" and "cap is" in r.error and slept == []


def test_rate_limit_retries_are_capped(tmp_path):
    r, _, slept = _throttled_run(tmp_path, failures=99)
    assert r.status == "error" and len(slept) == 6