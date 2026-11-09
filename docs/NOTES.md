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

(fill in during Step 6 to 8)