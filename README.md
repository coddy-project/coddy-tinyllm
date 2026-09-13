# coddy-tinyllm

A task-specific small language model that edits [Coddy](https://github.com/coddy-project/coddy-agent)'s
own configuration, locally, on a CPU, from a sentence in Russian or English.

Give it a `config.yaml` and an instruction. It decides whether the instruction touches the settings
at all; if it does not, it does nothing; if it does, it emits the `config_set` commands that carry it
out, verified against the live schema before anything is staged.

**Status: research and design.** No model has been trained yet. What exists is the problem
definition, a survey of what has actually been published, an architecture, a data plan, costed
hardware estimates, and a schema-corpus measurement tool that runs today.

## Why it might work

Coddy already ships the hard half. `internal/config/uci.go` defines a four-verb edit language
(`set`, `add_list`, `del_list`, `delete`) with a parser, a dry-run and a staging area, and
`internal/tools/config_*.go` exposes it to the agent. The config schema is 208 addressable fields,
~10 000 tokens of description, embedded in the binary and validated on every load. So the target
language is small, the verifier is free and exact, and the safety net (stage, review, commit,
rollback) is already built and already trusted.

What is missing is the cheap half: today "выключи телеграм-гейтвей" costs a full round trip to a
frontier model, and does not work at all offline.

## The three decisions that shape everything

**The model learns intent, not the schema.** Field names, types and defaults are retrieved from the
schema at runtime and copied; they are not in the weights. A Coddy release that adds a field works
on day one, with no retraining. Training enforces this by generating most examples against
*deliberately mutated* schemas, because the schema-guided-dialogue literature shows that models
otherwise memorise field names and the resulting zero-shot numbers are inflated.

**A grammar makes invalid output unrepresentable.** The decoder is constrained at request time to
the retrieved paths and their declared types. Syntax errors and invented field names stop being
possible; what remains is picking the wrong field, which is what the evaluation measures.

**It is allowed to give up.** Every output passes Coddy's own parser and dry-run, and anything that
fails twice escalates to the normal LLM path that exists today. The component is a fast path, never
a replacement, which is why it can ship long before it is excellent.

## What the cross-review changed

Three independent reviewers ([docs/08-review.md](docs/08-review.md)) landed on the same verdict, and
it moved the project's centre of gravity: **the deterministic pipeline is the product, and the model
is an optional optimisation that has to earn its place.** Retrieval over the schema, a compositional
parser for the twenty common intents, the grammar and the verifier are useful on their own, ship
first, and may well be the whole feature. Every milestone that costs money now sits behind a
measurement that can cancel it - including the first one, which counts whether anybody edits their
config often enough to justify any of this.

The same review caught three arithmetic errors, a metric stated backwards (a "no-op recall floor of
0.90" would have licensed editing the config on one in ten unrelated turns), an ambient suggestion
that could have ridden along with a later commit through Coddy's shared staging area, and a missing
security model for the fifth of the schema that is effectively code execution.

## What the research says, in four lines

- Sub-1B models do reach production quality on narrow structured tasks - 91 % full-pass on
  Kubernetes manifests from 1 000 examples at 1.5 B; 61 % to 98 % on tool calling from 5 000 examples
  at 350 M.
- Diff formats are dead below 3 B (0.5 B models score ~0.00 exact-match on every diff format tested),
  so the line-oriented DSL Coddy already has is the right target.
- Ternary at our size buys ~1.85× over Q4_0 on CPU, and on disk far less than the name suggests: the
  1.58 bits apply to the linear layers only, the embedding stays 4-8 bit, and the whole file lands at
  **2.4-3.7 bits per weight**. Distilling an FP16 model to 1.58 bit does land *on par with the
  teacher* on narrow tasks, so track B is a real option - just not a 4× one.
- The whole supervised programme costs **under $15 of rented GPU time**. The expensive parts are
  generating the data and, optionally, the ternary conversion (~$70-100).

## Layout

```
docs/01-problem.md        what the thing does, the contract with Coddy, ship criteria
docs/02-prior-art.md      the survey, with numbers and URLs, and the gaps nobody has filled
docs/03-architecture.md   prefilter, gate, retrieval, constrained generation, verification
docs/04-data.md           schema mutation, instruction backtranslation, donated sessions
docs/05-training.md       recipe, base-model licensing, VRAM and cost arithmetic, CPU-only training
docs/06-eval.md           the four-rung baseline ladder and the sets that can falsify the design
docs/07-roadmap.md        milestones, each with a kill criterion
docs/08-review.md         independent cross-review of this plan, and what it changed
notes/habr-1074678.md     the article that started this, and why it is a reference and not a base
notes/reviews/            the three reviewers' raw answers, unedited
experiments/schema_index/ builds the retrieval corpus from Coddy's schema and measures it
experiments/bitness.py    what bits-per-weight a "1.58-bit" model really lands at
experiments/cpu_training/ measures whether any of this can be trained without a GPU
```

## Try the one thing that runs

```bash
python3 experiments/schema_index/build_cards.py -o cards.jsonl
```

Reads Coddy's `config.schema.json` and reports the corpus the retriever will serve: 208 cards,
~9 900 tokens in total, ~48 tokens each - which is why retrieval is not an optimisation here but the
thing that makes a sub-second CPU path possible at all.

## Relationship to Coddy

Eventually `external/tinyconfig/` behind a build tag, in the same shape as `external/cli`,
`external/ui` and `external/swarm`: absent from the default build, stubbed in the untagged core,
documented under `docs/features/` like any other capability. Until then it lives here, where it can
be wrong cheaply.
