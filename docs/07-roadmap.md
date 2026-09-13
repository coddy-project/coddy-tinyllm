# 7. Roadmap

Sequenced so that the cheap, load-bearing questions are answered before the expensive, exciting ones.
Every milestone has a kill criterion, because the most likely failure of a project like this is not
that it does not work - it is that it works slightly worse than a thousand lines of ordinary code,
and nobody measures that until the weights are already in the binary.

## M0 - the floor, no neural network at all

*Deliverables:* the card generator (done - `experiments/schema_index/`), a Go retriever (BM25 +
static embeddings), ~20 hand-written intent rules, the first 300-example eval set, the metric
harness, all four rungs of [06-eval](06-eval.md) §6.1 wired up with R0 and R1 populated.

*Why first:* it produces the retrieval layer the model needs anyway, it produces the eval harness
everything is judged by, and it answers the only question that can cancel the project.

*Also in M0, and arguably the most important number in the project:* **measure the demand.** On this
machine 10 of 111 sessions so much as mention `config_set`. If operators edit their config twice a
month, a 150-400 MB artefact, a new runtime and six platforms of maintenance are being spent on
something a frontier call already handles at a cost nobody notices. M0 therefore ships the explicit
`coddy config "..."` entry point on top of R0 and counts its use before any model work is authorised.

**Kill criteria, two of them:** if R0 (rules + retrieval) lands within ~10 points of R1 (frontier
model with the same retrieval) on the real request distribution, ship R0 and stop. And if the
explicit entry point sees negligible use over a release cycle, stop regardless of how good the
numbers look - there is no bottleneck here to relieve.

The deterministic baseline is to be built seriously, not as a straw man: a compositional parser over
the cards (verb, field, value, unit conversion, aliases, current-value arithmetic) rather than one
rule per field. If this project is going to be beaten by ordinary code, better to find that out from
good ordinary code.

**What starts M1, stated as a rule rather than a hope:** M1 begins only when M0 has shipped, demand
has been counted, and the *residual* errors of the deterministic path form a pattern rules cannot
express - multi-clause instructions, indirect phrasing, cross-field reasoning. "R0 is a few points
below a frontier model" is not that rule: a frontier model finds this task trivial, so that
comparison is a bar good ordinary code clears, and clearing it proves nothing in either direction.
The default outcome of M0 is **ship R0 and train nothing.**

## M1 - the data engine and the ceiling

*Deliverables:* the generator of [04-data](04-data.md) (mutation, backtranslation, verifier
rejection, round-trip check), 5 000 verified examples, the R1 ceiling measured properly, and the
format experiment of [06-eval](06-eval.md) §6.4 settled.

**Kill criterion:** if R1 with perfect retrieval cannot clear 0.85 semantic equivalence, the task
definition is wrong (probably the DSL, probably ambiguity in the requests) and it gets fixed here,
before any GPU time is spent.

## M2 - the model

*Deliverables:* SFT of Qwen3-0.6B on 20-50 k examples, rejection-sampling round, evaluation against
the full ladder, then GRPO if and only if the no-op numbers need it. Cost: under $15 of rented GPU
(see [05-training](05-training.md) §5.3), days not weeks.

*Also here:* the 360 M variant, trained identically, to find out what capacity actually buys on this
task. The answer determines whether the ternary track is even interesting.

**Kill criterion:** `future-schema` within 10 points of `synthetic-held-out`. If not, the model is
memorising and the fix is architectural, not more data.

## M3 - the Go runtime spike

*Deliverables:* a standalone Go program that loads the GGUF, tokenises with parity against the
training tokenizer, generates under a GBNF grammar built at runtime from the schema, and reports p50
and p95 latency on x86 and arm64. Built with `CGO_ENABLED=0` and cross-compiled to all six Coddy
targets, or the milestone has failed.

*Unknowns being settled here:* whether [goccy/go-llama](https://github.com/goccy/go-llama) holds up
(three months old, so this is a spike, not an adoption), the `GOAMD64=v2` question, tokenizer parity,
and binary size.

**Fallback ladder, in order**, so the bet on a young dependency is bounded: (1) a cgo build tag over
llama.cpp for people who want it, default build unaffected; (2) a downloaded `llama-server` sidecar
that Coddy talks to over HTTP, which is how most tools ship this anyway; (3) an external
`coddy-tinyconfig` helper binary; (4) no on-device model, R0 rules only. Note that (1)-(3) all
weaken the "one static Go binary" pitch that makes track B attractive, which is why the two bets -
pure Go, and ternary - are kept separable and neither is allowed to justify the other.

**Kill criterion, stated without euphemism:** if no cgo-free Go path meets the latency budget, the
feature ships at rung (1) or (2), which means **the default build does not have it and the offline,
no-cloud promise is not delivered to the ordinary user**. The product in that case is M0 - retrieval,
the compositional parser, the grammar and the verifier - and the neural component becomes a follow-up
project with its own runtime. That is a real outcome, not a footnote, and it should be named out loud
rather than discovered later.

## M4 - integration

*Deliverables:* `external/tinyconfig/` behind a build tag, `Suggest()` wired in front of the existing
`config_set` staging path, an explicit entry point (`coddy config "..."` / a slash command) shipped
first, the ambient gate shipped second and off by default. Docs under `docs/features/`, a row in
`docs/reference/config.md`, the eval set running in CI - Coddy's documentation contract applies to
this like to anything else.

## M5 - ternary, optional

Only if M2 and M3 both landed, and only in this order: reimplement BitNet Distillation (nobody has
released the code), convert the M2 checkpoint, measure against R2 on the same ladder, and confirm
something can actually *run* it from Go. ~$70-100 of GPU and about a week of one desktop card.

**Kill criterion:** ternary must land within a few points of R2 *and* be runnable cgo-free, or it
stays a research note. The measured speed advantage over Q4_0 at this size is ~1.85×, not an order of
magnitude ([02-prior-art](02-prior-art.md) §B), so the honest justification for track B is size -
a ternary 360 M model is ~90-125 MB against ~210 MB at Q4, and the whole file is 2.4-3.7 bits per
weight rather than the 1.58 the name implies ([02-prior-art](02-prior-art.md), and the arithmetic in
[03-architecture](03-architecture.md) §3.3) - which only matters at all if the model is good enough
to be worth embedding.

## The artefact manifest

Three things version together and must be released together, or the whole design quietly breaks:
the **model**, the **schema** it was evaluated against, and the **alias file** the retriever uses.
A single manifest - digests of all three, plus the Coddy version range they are valid for - ships
with the weights and is checked at load. A model whose manifest does not match the running binary's
schema digest still works (that is the point of the design) but it logs the mismatch and its
evaluation numbers are known to be stale. Getting this wrong is how a "no retraining needed" system
turns into a system that silently regressed two releases ago and nobody noticed.

## Open questions

Written down so they are answered deliberately rather than by accident:

1. **Ambient or explicit?** Running the gate on every turn is the feature people will actually
   notice; it is also the one that can misfire. Ship explicit first, gather the no-op numbers from
   real use, then decide.
2. **Where do the weights live?** Embedded (offline, zero setup, fattens every release archive
   across six targets) or downloaded on first use (the consensus in the Go ecosystem, adds a network
   dependency and a failure mode). Ternary makes embedding defensible; Q4 does not.
3. **Does the alias file need a big model at all?** If retrieval over descriptions alone reaches
   recall@12 above ~0.95, the alias generation step disappears, and with it one moving part per
   release.
4. **How far does this generalise beyond config?** The same architecture - retrieve candidates,
   constrain the grammar, verify, abstain - would serve slash commands, skill selection and MCP
   server management. Worth keeping in the interface design, worth resisting until the first one
   works.
5. **Does the gate belong in Coddy proper?** A 5 MB "is this about settings" classifier is useful to
   the main agent too, as a routing hint, independently of whether the generator ever ships.
