# Week 1 notes

## Smoke test results

| Task | Provider | Passed | Status | Turns | Cost | What went wrong or surprised me |
|------|----------|--------|--------|-------|------|--------------------------------|
| fix_bug | | | | | | |
| implement | | | | | | |
| rename | | | | | | |
| add_flag | | | | | | |
| ambiguous | | | | | | |

## Bugs found and fixed

- Verified with scratch/check_claim.py: for [[1], [1]] the original returns [[1]] and the
  model's Counter version raises TypeError (unhashable). The model's final message claimed
  "behavior is unchanged". The visible tests can't detect this.


## Day 6: trace investigations (9 traces, one open model on Groq: qwen/qwen3.8-27b)

### ambiguous (before fixes): completed, 12 turns, 199.8s, 23,488 input tokens
- First bad turn: 5. A multi-line `python -c` returned empty output with exit code 0, and the
  harness labelled it [ok]. Turns 5-8 (4 of 12) were spent working around it; turn 9-10 wrote
  bench.py and it worked.
- Cause: cmd.exe does not handle line breaks in a command (confirmed with
  scratch/cmd_newline.py: output for the line-break case matched an empty program: ___).
- Fixes: bash rejects line breaks on Windows with a clear error; empty output is labelled
  "[no output]"; the Windows prompt hint says to write multi-line code to a file.
  Tests: test_bash_rejects_line_breaks_on_windows, test_bash_labels_empty_output.
- After the fixes (3 trials): turns = ___, ___, ___
- Overclaim: the final message said behavior was unchanged. Counter needs hashable items, so
  lists crash it; the old version handled them. The visible tests can't see this, so "passed"
  does not mean "behavior preserved". Plan: hidden tests in week 4.

### fix_bug x2, implement, rename, add_flag
- No tool errors, no bad JSON, no retries. fix_bug twice gave the same tool sequence
  (6,168 and 6,162 input tokens). Tasks may be too easy to separate models (4-6 turns).
- Independent pass column (from results JSON): ___

### Latency
- Same turn-1 request (about 1,050 in / 27 out): 0.6s in fix_bug vs 7.5s in add_flag. 27
  output tokens cannot take 7.5s to generate, so the difference was waiting.
- SDK retries were hidden. Now logged as transient_retry events.
- After the change: waits in the full pass = ___ s across ___ retries. Conclusion: ___
- Groq limits for this model (from the console): ___ requests/min, ___ tokens/min, ___ tokens/day

### Anthropic / OpenAI
- Not tested live (no credits). The adapters are unit-tested with fake clients only.
- The one CLI run failed: 400 credit balance too low. The loop ended cleanly (error, 0 turns).
  The CLI defaulted to anthropic; now it defaults to HARNESS_PROVIDER or the first key found.

### Bugs found and fixed
- Line endings (edit_file/write_file used text mode) -> bytes I/O. 3 tests.
- Silent empty bash output / Windows line breaks -> see above. 2 tests.
- Hidden SDK retries -> loop-owned retries with traces. 5 tests.

### Open problems (roadmap)
- History grows every turn -> week 3: context management (also lowers tokens per minute)
- bash runs on my real machine -> week 2: sandbox
- One run per task -> week 4: multiple trials, harder tasks, hidden behavior tests
- Free-tier rate limits -> week 4: runner that spaces calls and can resume


(fill in during Step 6 to 8)