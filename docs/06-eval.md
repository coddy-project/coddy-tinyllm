# 6. Evaluation

The eval harness is built before the model, because the decision this project has to keep making -
is the model earning its bytes? - is only answerable against a fixed ladder of baselines.

## 6.1 The ladder

Every number gets reported for all four rungs, on the same sets:

| Rung | What it is | Why |
|---|---|---|
| R0 | BM25 over field cards + ~20 hand-written intent rules, no neural net | the floor. If the model cannot clear this by a wide margin, the project is a retriever |
| R1 | retrieval + grammar + a **frontier model** | the ceiling. Anything R1 cannot do is a data or task-definition problem, not a model-size problem |
| **R1s** | retrieval + grammar + an **off-the-shelf Qwen3-0.6B/1.7B**, prompted, not fine-tuned | **the rung that decides whether to train at all** |
| R2 | retrieval + grammar + our fine-tuned small model | the deliverable |
| R3 | R2 with the ternary weights | track B, only if it lands within a few points of R2 |

R1 matters more than it looks: if a frontier model with the same retrieval and grammar scores 0.72
exact-match, then the retrieval or the task definition is the bottleneck and no amount of fine-tuning
a 600 M model will fix it.

**R1s matters most of all**, and it is deliberately placed before any GPU time is spent. The grammar
already guarantees validity, the retrieval already supplies the field names, and the task is a
four-verb DSL - so it is entirely possible that a stock Qwen3-0.6B with a good prompt clears the ship
floor and the entire training programme is unnecessary. That would be an excellent outcome and the
plan should be built to discover it early rather than to avoid discovering it. Fine-tuning is
justified only by the measured gap between R1s and R2 on the *real* request distribution, not on
synthetic paraphrases of our own generator's output.

## 6.2 Metrics

**Primary - semantic equivalence.** Apply the emitted commands to the input config, apply the gold
commands to the same input, compare the resulting documents. Deterministic, cheap, and it forgives
irrelevant differences in command order or phrasing that exact match punishes. This is the headline
number.

**Exact command-set match** after canonicalisation (sorted, paths normalised, values typed). Brittle,
but it is the one that correlates with a clean diff in front of the operator.

**Schema validity rate** of the post-edit config. With a grammar in place this should be 1.0 by
construction, so it functions as an assertion rather than a metric - if it drops below 1.0 the
grammar has a hole.

**Valid-but-wrong rate**, reported separately and treated as the primary hazard. Once a grammar is in
place, a parse failure is a non-event - it is caught, it costs a retry, nobody is harmed. The
dangerous output is a well-formed, schema-valid, dry-run-clean command that changes the wrong field,
because every mechanical check passes and only the operator is left to notice. Evaluation must count
these separately from failures rather than folding both into one accuracy number, and the operator-
facing diff is the last line of defence, which is a UX requirement as much as a model one.

**No-op confusion matrix**, reported separately for Russian and English, and separately for the two
no-op classes (not about config / already in that state). Precision here is the ship gate.

**Retrieval recall@k** (is the gold field in the top k) at k = 5, 12, 25, borrowed straight from the
text-to-SQL schema-linking literature. Retrieval recall is a hard ceiling on everything downstream:
the generator cannot emit what it was never shown - and on the `future-schema` set it is a **ship
gate in its own right**, held to the same bar as generator accuracy. A field added in a release whose
description is one terse line is a retrieval failure long before it is a generation failure, and that
is the realistic way the no-retraining promise breaks.

**Online signal**, once anything ships: the rate at which an operator rejects or edits a staged
suggestion, logged locally. It is the only metric drawn from the true distribution, and it is the
early warning that the offline sets have drifted from reality.

This needs a mechanism, not an intention. Every decision writes one structured local log line -
terminal state, confidence, retrieved field paths, emitted commands, verification result, whether the
operator accepted - and `coddy` can print the resulting confusion matrix on demand. Without it the
promise that the component "is not allowed to be silently wrong" is unenforceable, because nothing
would be counting. The log stays local and carries no values, only paths, for the same reason the
donated-dataset path redacts.

**Latency** at p50 and p95, on 4 threads, measured on three machines: a modern x86 laptop, an older
x86 laptop (the development machine here is an i7-8750H, which is a fair floor), and an arm64 Mac.

**Artefact size**: bytes added to the binary, bytes added to `~/.coddy`, cold-start time.

## 6.3 Evaluation sets

| Set | Size | Source | What it measures |
|---|---|---|---|
| `synthetic-held-out` | 2 000 | same generator, different seed and different teacher | in-distribution competence |
| `mutated-schema` | 1 000 | schemas mutated more aggressively than training | resistance to memorisation |
| `future-schema` | 300 | a **real later Coddy release**, generated after the training snapshot | the load-bearing claim of the whole design |
| `real-donated` | 100-300 | operator-donated sessions, redacted, eval-only | the real request distribution |
| `adversarial-noop` | 1 000 | code, git, shell and debugging requests that mention config words | no-op precision |
| `ambiguous` | 300 | requests a careful human would ask back about | abstention behaviour |
| `ru-hard` | 500 | colloquial, mistyped, mixed-script Russian | the failure mode most likely to be invisible to an English-speaking reviewer |

`future-schema` is the set that cannot be faked. It is generated only after a Coddy release that
adds or renames fields, against a model frozen before it. If the gap between `synthetic-held-out` and
`future-schema` exceeds ~10 points, the copy-don't-recall design failed and no amount of additional
training data will fix it - the fix would be architectural (index-anchored output, see
[03-architecture](03-architecture.md) §3.2).

## 6.4 The format experiment

Because nobody has published it, and because it determines the output language, milestone M1 runs a
controlled comparison on the same data with the same base model: UCI DSL against JSON Patch
(RFC 6902) against JSON Merge Patch against whole-file YAML rewrite. Metric: semantic equivalence and
output token count. The expectation from [Diff-XYZ](https://arxiv.org/html/2510.12487v2) is that the
DSL wins by a wide margin at this size and whole-file rewrite is unusable on long configs, but
"expected" is not "measured", and this is cheap to settle.

## 6.5 Harness shape

Two halves, deliberately sharing the gold data:

- **Python**, for the training loop: standard `datasets` + a scorer that shells out to the Go
  verifier. Optionally wrapped as an
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) plugin metric so the
  numbers sit next to standard task scores.
- **Go**, for the shipped path: a table-driven test in the Coddy tree that runs the eval set through
  the real `Suggest()` entry point and the real `internal/config` verifier, and fails CI on
  regression against a checked-in baseline. This is the one that keeps the feature honest after the
  research is over.

The gold data lives in this repository as JSONL, versioned, with the schema snapshot it was generated
against, so a result from six months ago stays reproducible when Coddy's config surface has moved on.
