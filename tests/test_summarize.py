from harness.summarize import summarize
from harness.trace import Tracer

USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0, "cache_write_tokens": 0}


def test_summarize_reads_a_trace(tmp_path):
    path = tmp_path / "t.jsonl"
    t = Tracer(path, "r")
    t.emit("run_start", task="demo", provider="fake", model="m")
    t.emit("model_response", turn=1, text="", tool_calls=[{"name": "glob"}],
           usage=USAGE, latency_s=0.5)
    t.emit("tool_result", turn=1, name="glob", is_error=True, content="boom")
    t.emit("run_end", status="completed", turns=1, cost_usd=None, wall_s=1.0,
           error=None, usage=USAGE)
    t.close()
    out = summarize(path)
    assert "glob (ERR)" in out and "status=completed" in out and "cost=unknown" in out