# Evaluation results

Latest run: **September 25, 2026**, 10:51:18 to 10:56:44 UTC.
These measurements used **GPT-4.1** and **Jev 1.13.0**.

[Saved results](results/2026-09-25.json) include the scores, failed case IDs,
conversation summaries, run timestamps, and hashes of the evaluated source files.
At the time of that run, the local test suite passed 92 tests, plus lint and
whitespace checks. This is a historical count, not the current test-suite size.

## Verifier comparisons: `compare_verifiers`

This command gives both verifiers the same candidate message and prepared fact.
It tests whether they accept or reject that fact correctly. It does **not** run
extraction or a complete conversation.

| Suite | What it covers | Cases | Requests per verifier |
| --- | --- | ---: | ---: |
| `benchmark` | Development examples, including ambiguity, misleading statements, and dates | 120 | 180 |
| `holdout` | A separate fixed set of examples | 50 | 80 |
| `corrections` | Genuine correction requests and invalid requests | 7 | 15 |

High-risk cases receive two extra runs. Accuracy counts each original case once;
the repeated runs measure whether the verdict changes.

### Benchmark

| Measure | GPT-4.1 | Jev |
| --- | ---: | ---: |
| Correct cases (higher is better) | 92/120 | 108/120 |
| False accepts (lower is better) | 26 | 11 |
| Inconsistent cases (lower is better) | 1 | 0 |

### Holdout

| Measure | GPT-4.1 | Jev |
| --- | ---: | ---: |
| Correct cases (higher is better) | 37/50 | 48/50 |
| False accepts (lower is better) | 11 | 2 |
| Inconsistent cases (lower is better) | 4 | 0 |

### Corrections

| Measure | GPT-4.1 | Jev |
| --- | ---: | ---: |
| Correct cases (higher is better) | 5/7 | 7/7 |
| False accepts (lower is better) | 2 | 0 |
| Inconsistent cases (lower is better) | 0 | 0 |

A **false accept** means accepting an unsupported fact. Neither verifier falsely
rejected a supported fact in these suites. Correct-case totals also include
judgments about the type of reply, so they cannot be calculated from false accepts
alone. Inconsistent cases are those whose verdict changed across repeated runs.

Jev correctly handled all seven correction cases. GPT-4.1's two failures were:

- Treating an unspecified complaint about the summary as a salary correction.
- Accepting a command to save an invented salary as a factual assertion.

Both verifiers still accepted incorrect date interpretations in the benchmark.
Higher overall scores do not remove the need for application checks and candidate
confirmation.

## Complete conversations: `conversations --verifier provider,jev`

This command runs the whole intake: extraction, verification, follow-up questions,
corrections, and the final record. GPT-4.1 extracts facts in both configurations.
The columns below identify **which verifier** controls the conversation.

| Scenario | GPT-4.1 verification | Jev verification |
| --- | --- | --- |
| Answer each question separately | Passed, 6 turns | Passed, 6 turns |
| Volunteer all details early | Passed, 2 turns | Passed, 2 turns |
| Correct salary during review | Passed, 8 turns | Passed, 8 turns |

Both configurations completed all three scenarios with correct final records,
no repeated answers, and no comparison-request errors.

When multiple verifiers are selected, identical conversation inputs reuse the same extraction. Each turn
also records the other verifier's judgment of those exact facts. Each verifier
still drives its own conversation, so later questions can differ. The evaluator
refuses to confirm a wrong final record.

These runs include additional comparison calls and cached extractions, so their
elapsed times are not a fair comparison of normal application latency.

## How to interpret these results

The fixes passed the three scripted conversation scenarios, and Jev performed
better on the isolated verifier suites in this run. Neither result establishes
reliability across real candidate conversations: both verifiers still made false
accepts, and the conversation suite contains only three scenarios.

The examples and expected answers are synthetic. The holdout was originally
reserved for a separate check but has now been run repeatedly; this is not a
fresh, unseen validation set. Model outputs can vary between runs.

All run commands and credential requirements are in the repository [README](https://github.com/RomansBermans/llm-data-intake/blob/main/README.md).
