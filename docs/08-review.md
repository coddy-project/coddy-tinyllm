# 8. Cross-review

The plan in this repository was reviewed by three independent agents, each given the design documents
inline and no access to this conversation. Every finding below was then verified against the Coddy
source or the arithmetic before being accepted; the verdicts are mine, not theirs.

| Reviewer | Model | Outcome |
|---|---|---|
| Codex | `gpt-5.6-sol`, reasoning effort high | 17 findings, verdict "build a reduced version" |
| Cursor Agent | `auto` | 7 findings, verdict "do not build the custom-trained model yet, build the reduced version" |
| Coddy | `neuraldeep/qwen3.8-27b-noreason`, self-hosted | 7 findings, verdict "build a reduced version - the rules engine is the product" |

(The reasoning variant of the same self-hosted model returned nothing in ten minutes on the same
brief; the non-reasoning variant answered in about four. Worth remembering for the next review.)

All three arrived independently at the same verdict, and at four of the same findings. That agreement
is the most useful output of the exercise, and it is a criticism of the plan's centre of gravity
rather than of its parts. The sharpest formulation came from the smallest reviewer:

> the plan is a well-researched justification for building a rules engine that is dressed up as a
> machine-learning project, and the rules engine is the part that should ship.

## Confirmed, verified, and fixed

**1. The decisive baseline was missing.** Both reviewers, independently: the ladder had a frontier
model (R1) and a fine-tuned small model (R2), but not **a stock Qwen3-0.6B/1.7B given the same
retrieval and the same grammar**. Since the grammar supplies validity and the retrieval supplies the
field names, it is entirely possible that no fine-tuning is needed at all - and the plan as written
would not have found out until after paying for the data.
*Verified:* correct by inspection. *Fixed:* rung **R1s** added to [06-eval](06-eval.md) §6.1, and the
training milestone is now gated on a measured R1s-to-R2 gap on human-written data.

**2. The no-op criterion was stated backwards.** With NOOP as the positive class, an unwanted edit on
a non-config turn is a NOOP *false negative*, so it damages **recall**, not precision - and the
"floor 0.90 recall" in the first draft would have licensed editing the config on one in ten unrelated
turns.
*Verified:* worked through the confusion matrix; the reviewer is right and the original table was
dangerous. *Fixed:* [01-problem](01-problem.md) now states **`false_edit_rate = P(EDIT | not a config
turn)`** directly, at 1e-2 for explicit mode and 1e-4 for ambient, with the sample-size consequence
spelled out (rule of three: ~30 000 negatives for a 95 % upper bound at 1e-4).

**3. An ambient suggestion could ride along with a later commit.** Coddy's staging is per-session and
persistent, and `config_commit` commits everything staged.
*Verified in source:* `internal/tools/config_staging.go` writes `config_staging.json` into the session
directory precisely so pending commands survive restarts. An unasked-for suggestion staged at 14:00
would be committed with the operator's own edit at 14:20. *Fixed:* [03-architecture](03-architecture.md)
§3.4 - ambient mode never touches shared staging; it produces an isolated proposal carrying the source
turn, a hash of the config it was computed against, the diff and an expiry, staged only on explicit
acceptance and invalidated if the config moved.

**4. The security model was absent.** Neither prompt injection nor the fact that some config fields
are code execution appeared in the first draft.
*Verified by enumerating the schema:* 39 of 208 fields are security-relevant, and `mcp_servers[].command`,
`mcp_servers[].args`, `providers[].api_key_command`, `hooks.files`, `hooks.enable`,
`tools.command_allowlist`, `skills.dirs` and `subagents.dirs` are process execution or its
prerequisites, while `providers[].api_base` redirects every request and its credentials. *Fixed:*
[03-architecture](03-architecture.md) §3.6 - a green/amber/red writable-surface policy enforced in Go,
red-class paths absent from the grammar entirely, and the rule that the fast path reads **only the
operator's own typed instruction**, never tool output, file contents or a fetched page.

**5. The grammar guarantee was overstated.** A trie over paths with coarse types cannot enforce
ranges, patterns, formats, nullability, union members, list-element structure or cross-field
invariants.
*Verified, with independent corroboration:* the best pure-Go JSON-Schema-to-GBNF port
(`ThiraSoft/golem`) explicitly **refuses** `pattern` and `minimum`/`maximum` rather than pretending to
support them. *Fixed:* the claim is narrowed to "syntactically malformed and out-of-candidate paths
are not representable", with the dry-run and schema check named as non-optional.

**6. Selector paths are not schema paths.** `mcp_servers[name=context7].command` contains a value
that comes from the current document, not from the schema.
*Verified in source:* `parseDottedConfigPath` in `internal/config/uci.go` treats the selector as part
of the segment; nothing in the schema enumerates the legal values. *Fixed:* the path trie is built
from schema paths **instantiated against the live config**, with creating a new entry as a separate
enumerated alternative.

**7. NOOP conflated four different control flows.** "Not about settings" returns control to the
agent; "already in that state" completes without an edit; "ambiguous" asks; "unsupported" escalates.
*Fixed:* five terminal states (`EDIT`, `NOT_CONFIG`, `ALREADY_SATISFIED`, `AMBIGUOUS`, `UNSUPPORTED`),
with confidence derived from retrieval margins and OOD signal rather than from a self-reported token.

**8. Three arithmetic errors.**
- rejection sampling was priced at "~400 M output tokens"; it is **12 M output tokens**
  (50 k × 8 × 30) plus 480 M prefill without prefix reuse, 60 M with;
- "100 B tokens is a week on 8×H100" contradicted the FLOPs in the same section: 1.56e20 / 2.7e15 is
  **16 hours**;
- the ternary memory advantage was written as 4×; TQ2_0 at 2.0625 bpw against Q4_K_M at ~4.5-4.8
  effective bpw is **2.2-2.3×**.
*All three verified and fixed* in [05-training](05-training.md) and [03-architecture](03-architecture.md).

**9. The size budget was incompatible with the model.** A 0.6 B model at the 1.58-bit floor is 118 MB
and at TQ2_0 is ~155 MB, before tokenizer, scales and the non-ternary embedding and output layers -
so the original 100 MB target could never have been met by a 0.6 B ternary model.
*Fixed:* budget restated at 150 MB, and the tension made explicit - **ternary and "embedded in the
binary" only coincide at 300 M or below**, which is also where the Spectra scaling law hurts most.
That tension is now stated as the actual content of track B.

**10. Session negatives are positive-unlabeled.** A turn that did not lead to a config edit may be a
turn where the agent missed the intent, took another route, or was interrupted.
*Verified:* on this machine 10 of 111 sessions mention `config_set` and most of those mentions are in
reasoning prose rather than a tool call - the label noise is real and large. *Fixed:*
[04-data](04-data.md) §4.1 now treats them as PU data requiring a stratified audit.

**11. Demand was never measured.** A 150-400 MB artefact, a new runtime and six platforms of
maintenance may be optimising an operation people perform twice a month.
*Fixed:* M0 in [07-roadmap](07-roadmap.md) now ships the explicit entry point on the deterministic
baseline and **counts its use**, with negligible use as a kill criterion independent of model quality.

**12. The verifier was doing less work than it could.** Three checks moved out of the model and into
deterministic code, all from the Coddy reviewer: **already-satisfied** (compare the post-edit document
against the current one - if nothing changes, the answer is `ALREADY_SATISFIED` whatever the model
believed, which removes the hardest abstention class from the model entirely); a **semantic
plausibility score** (does a number in the instruction match the value, do the field's segments and
description share tokens with the instruction) that escalates a high-surprise edit to explicit
confirmation and is the only thing standing between "make the tests run faster" and
`set agent.max_turns=4`; and a **five-command cap**, because a diff nobody reads is not a safety
mechanism.

**13. The "no retraining" promise lives in the alias file, not in the model.** Retrieval recall on a
brand-new field caps everything downstream - a perfect generator with 0.85 recall on new fields scores
0.85 - and recall on new fields comes from the per-release alias file.
*Fixed:* alias regeneration is now a **release-blocking step**, in the same category as a schema
migration, and retrieval recall@12 on the future schema is a ship gate. A stale alias file does not
fail loudly; it quietly moves the system below its own floor.

**14. Nothing was counting.** The plan asserted "allowed to be wrong, not allowed to be silently
wrong" with no mechanism behind it.
*Fixed:* one structured local log line per decision - terminal state, confidence, retrieved paths,
emitted commands, verification result, operator acceptance - and a confusion matrix on demand. Paths
only, never values, for the same reason the donated-dataset path redacts.

**15. Smaller corrections, all applied:** the cost headline now leads with data ($200-500), not GPU
rent (<$15); GRPO and rejection-sampling estimates are marked soft and require a measured pilot;
BitDistill's 10 B-token warm-up is flagged as a general-model figure to ablate (1 B / 3 B / 10 B)
rather than pay by default; the deterministic baseline is specified as a compositional parser rather
than "about twenty rules"; retrieval recall@12 on the future schema becomes a ship gate in its own
right; training gains deliberate hard slices (gold buried at rank 8-12, near-synonym competition,
selector construction); a model/schema/alias **artefact manifest** is required; the runtime fallback
ladder is written down explicitly; and an online signal (operator rejects a staged suggestion) is
added to the eval.

**16. Bilingual coverage was a footnote.** A single aggregate number hides a Russian false-edit rate
twice the English one. *Fixed:* a held 50/50 split in the data, and per-language floors in the ship
criteria rather than an aggregate.

## Found while verifying, not by a reviewer

Checking the latency table against the arithmetic turned up the worst problem in the original plan.
This workload is **prompt-heavy and output-light** - ~1 000 tokens in, ~30 out - while every published
tokens/second figure describes *generation*. Prefill costs ~`2·N·T` FLOPs: ~8.8e11 for a 0.6 B model
and 1 000 tokens, against something like 50-150 GFLOPS from four AVX2 threads on quantised GEMM. That
is **seconds, not the 300-800 ms the first draft claimed**, and the "sub-second fast path" premise
does not survive it without changes.

Recorded in [03-architecture](03-architecture.md) §3.5 as an unmeasured 1-5 s risk that M3 measures
before anything is trained, with the mitigation ladder (fewer and shorter cards, a smaller generator,
prefix reuse) - and with §3.7, which is the real consequence: the task decomposes into classification,
ranking and span extraction, all of which are encoder-shaped and two orders of magnitude cheaper than
a decoder's prefill. A cascade of small encoders with the generative model handling only the residue
may be the version of this project that actually ships, and it resolves the artefact-size question
without ternary having to.

## Accepted in part

**"Mutating the schema tests resistance to memorisation, not the ability to learn a new concept."**
Correct, and the strongest intellectual criticism of the design. Renaming `max_turns` to `turn_limit`
proves the model is not reciting; it does not prove the model can pick up a field whose *concept* it
has never seen from one line of description. Accepted: the claim is softened from "works on day one"
to "can address new fields without retraining", whole semantic families are held out rather than
individual names, and historical forward-chaining across real Coddy releases is the honest test.

**"Cut the ternary track."** Both reviewers said so, and on their information they were right. One
fact arrived after the brief was written and changes the calculus: `TQ1_0` and `TQ2_0` have been in
mainline llama.cpp since 2024, and the candidate Go runtime's own kernel index covers both - so track
B needs **no bespoke kernels and no second runtime**, which was most of the cost they were objecting
to. What survives of their objection, and is accepted: the speed gain at this size is ~1.85× rather
than an order of magnitude, ternary matmul is not the whole inference workload, BitNet Distillation
has no released implementation, and none of it matters until a Q4 model has proven demand. Track B
stays as M5 with kill criteria, off the critical path, and is not allowed to justify any other
decision.

**"Cut the separate gate model."** Kept as a design element, deferred as a build item. The published
evidence for a separate classifier over asking the generator is good, but it only pays for itself in
ambient mode, and ambient mode is now post-M4. Note the tension with §3.7: if the encoder cascade wins,
the gate stops being a separate artefact and becomes the first head of the only artefact.

**"M0 is a gate, not a milestone; the default outcome is ship-R0-and-train-nothing."** Accepted in
substance. The comparison was also wrong: R0 was to be judged against R1, a frontier model that finds
this task trivial, which is a bar a decent parser clears - so passing it would have proved nothing
either way. The roadmap now judges R0 against the *residual error distribution* (are the failures a
learnable pattern rules cannot express?) and against measured demand, and M1 does not start until
both say yes.

## Not accepted

**"Prefer index-anchored output (`set #1=40`) for the shipped model."** The grammar already makes an
out-of-candidate path unrepresentable, so index anchoring buys output tokens and costs readability in
logs, tests, diffs and the existing tooling. It stays where it was - in reserve for the smallest model
tier, where every token counts.

## What this changed overall

The plan did not lose a component; it lost its centre of gravity. Before review it read as "train a
small model, then build the scaffolding around it". After review it reads as "build the deterministic
feature, measure whether anyone wants it, measure whether a stock model already does the job, and
train something only if both answers say so". Every milestone that costs money now sits behind a
measurement that can cancel it.

The parts both reviewers called sound, recorded so they are not re-litigated later: the fast-path
contract with escalation, targeting Coddy's existing UCI language rather than a diff format, the
verifier as a reward, keeping the schema out of the weights, licence-filtering the base models, and
explicit-before-ambient deployment.
