<!-- Verbatim snapshot of the brief the three reviewers were given, kept for provenance.
     It concatenates the docs as they stood before the review, so its relative links
     point at docs/ and do not resolve from here. The current docs are the source. -->

# Cross-review request: design proposal, not code

You are reviewing a DESIGN PROPOSAL for a new side project. Everything you need is inline below.
Do NOT use any tools, do NOT try to read files or run shell commands, do NOT ask questions. Answer
in ONE message.

## The project in one paragraph

Coddy is an open-source coding agent written in Go (CLI + HTTP server + web UI, built with build
tags). It already lets an LLM edit its own YAML config through four UCI-style commands
(set / add_list / del_list / delete) with a parser, a dry-run, a staging area and a commit/rollback
flow. The proposal is to train a task-specific SMALL language model (roughly 300M-600M parameters,
possibly ternary/1.58-bit) that runs on CPU inside the Go binary behind a build tag, and that turns
a one-sentence instruction in Russian or English into those commands - or into "no config change
needed". The config schema changes with every Coddy release and the model must NOT be retrained
when it does.

## What I want from you

Ruthless, specific criticism of the PLAN. Assume the author is competent and does not need
encouragement. In particular, challenge these, and say so plainly when a part is actually fine:

1. Is training a custom model justified at all, versus (a) a rules engine plus retrieval, or (b) an
   off-the-shelf Qwen3-0.6B/1.7B with a good prompt and a grammar, no fine-tuning?
2. Is "retrieve schema slices + constrain the decoder + train on mutated schemas" actually enough to
   generalise to config fields that did not exist at training time? What breaks?
3. Is the ternary / 1.58-bit track worth doing, or is it hype given the measured ~1.85x speedup over
   Q4_0 at this size?
4. Are the hardware and cost estimates plausible? Point at any arithmetic that looks wrong.
5. Is the no-op / abstention design sound? This is the part that can silently corrupt someone's
   config.
6. The Go inference story depends on a three-month-old project (goccy/go-llama, llama.cpp
   transpiled from wasm to Go source). How bad is that bet and what is the fallback?
7. What is MISSING from the plan entirely?

Format: numbered findings, each with a severity (blocker / major / minor / nit), one-sentence claim,
the reasoning, and a concrete recommendation. End with a one-paragraph verdict: build it, build a
reduced version, or do not build it. Be concrete about which parts to cut.

---

# coddy-2bit-llm

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

## What the research says, in four lines

- Sub-1B models do reach production quality on narrow structured tasks - 91 % full-pass on
  Kubernetes manifests from 1 000 examples at 1.5 B; 61 % to 98 % on tool calling from 5 000 examples
  at 350 M.
- Diff formats are dead below 3 B (0.5 B models score ~0.00 exact-match on every diff format tested),
  so the line-oriented DSL Coddy already has is the right target.
- Ternary at our size buys ~1.85× over Q4_0 on CPU and ~4× on memory, not the 6× headline - but
  distilling an FP16 model to 1.58 bit lands *on par with the teacher* on narrow tasks, so track B is
  a real option rather than a stunt.
- The whole supervised programme costs **under $15 of rented GPU time**. The expensive parts are
  generating the data and, optionally, the ternary conversion (~$70-100).

## Layout

```
docs/01-problem.md        what the thing does, the contract with Coddy, ship criteria
docs/02-prior-art.md      the survey, with numbers and URLs, and the gaps nobody has filled
docs/03-architecture.md   prefilter, gate, retrieval, constrained generation, verification
docs/04-data.md           schema mutation, instruction backtranslation, donated sessions
docs/05-training.md       recipe, base-model licensing, VRAM and cost arithmetic
docs/06-eval.md           the four-rung baseline ladder and the sets that can falsify the design
docs/07-roadmap.md        milestones, each with a kill criterion
docs/08-review.md         independent cross-review of this plan, and what it changed
notes/habr-1074678.md     the article that started this, and why it is a reference and not a base
experiments/schema_index/ builds the retrieval corpus from Coddy's schema and measures it
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
# 1. The problem

## One sentence

Given a Coddy config file and a free-form instruction in Russian or English, decide whether the
instruction touches the configuration at all, and if it does, emit the `config_set` command list
that carries it out - locally, on a CPU, in under a second, without calling a cloud model.

## Why this is worth building

Coddy already has the whole machinery for a machine to edit its own config. `internal/tools`
carries `config_get`, `config_set`, `config_changes`, `config_commit`, `config_revert`,
`config_rollback` and a staging area, and `internal/config/uci.go` defines a four-verb command
language with a parser and a dry-run:

```
set      <path>=<value>
add_list <path>=<value>
del_list <path>=<value>
delete   <path>
```

Paths are dotted, with `name[key=value]` selectors for list entries (`mcp_servers[name=context7].command`).
Nothing is applied until `config_commit`, and a commit keeps a snapshot that `config_rollback`
restores.

What is missing is the cheap half. Today "turn off the Telegram gateway" costs a full round trip to
a frontier model, with the whole tool catalogue and system prompt in the prefix, and it does not
work at all when the operator has no key, no network, or is on the free tier of a provider that is
currently rate-limiting them. The task itself is narrow: a few hundred addressable fields, four
verbs, values that are mostly booleans, integers and short enums.

That is exactly the shape of problem a task-specific small model is good at, and exactly the shape
of problem a frontier model is a waste on.

## The contract

The component is a **fast path in front of an existing, already-safe tool**, never a replacement
for the agent.

```
instruction + config + schema
        |
        v
  [ gate ] --------- not a config request ---> return NOOP, agent continues as usual
        |
        v
  [ retrieve ] -> candidate fields from the live schema
        |
        v
  [ generate ] -> UCI command lines, grammar-constrained
        |
        v
  [ verify ] -> ParseUCICommands + DryRunUCICommands + schema check
        |            |
        |            +--- fails twice ---> escalate to the normal LLM path
        v
  staged edits, shown to the operator, committed only by config_commit
```

Three properties fall out of that and they are what make the project shippable:

- it is allowed to be wrong, it is not allowed to be silently wrong - every output passes the same
  parser and dry-run the big model's output passes, and the operator still confirms the commit;
- it is allowed to give up - an abstention costs one escalation to the path that exists today;
- it never invents a field name - the decoder is constrained to paths that exist in the schema
  loaded at runtime.

## The final case, stated precisely

Input:

- `config.yaml` as it is on disk (a real one is ~900 tokens; power users reach several thousand);
- one instruction, Russian or English, one to three sentences;
- `config.schema.json`, which is already embedded in the Coddy binary and changes with every release.

Output, one of:

- `NOOP` with a reason class (not about settings / already in that state / needs a human);
- an ordered list of UCI commands that parse, dry-run clean and validate against the schema.

Hard requirement: **the schema changes without retraining the model.** A new Coddy release adds
`swarm.rings[].fanout`, and the same model file must handle a request about it on day one.

## Non-goals

- Not a chat model. One turn in, one command list out, no dialogue, no explanations.
- Not a code editor and not a shell. "Primitive commands" here means the config verbs, plus
  possibly a small closed set of local actions later (see [07-roadmap](07-roadmap.md)); it does not
  mean free-form `run_command`.
- No multi-hop reasoning, no planning, no tool loops.
- Not a replacement for the agent when the request is ambiguous - ambiguity is an abstention.

## Success criteria

Measured on a held-out set built the way [06-eval](06-eval.md) describes:

| Metric | Floor to ship | Target |
|---|---|---|
| No-op precision (says NOOP when the turn is not about config) | 0.98 | 0.995 |
| No-op recall | 0.90 | 0.97 |
| Exact command-set match on config requests | 0.70 | 0.85 |
| Schema-valid output when it does emit commands | 0.99 | 1.0 |
| Unseen-field generalisation (fields added after the training cut) | within 10 pts of seen fields | within 5 |
| Latency, gate only, 4 CPU threads | 150 ms | 50 ms |
| Latency, full generation, 4 CPU threads | 1.5 s | 500 ms |
| Artefact size added to the binary or `~/.coddy` | 400 MB | 100 MB |

No-op precision is the one that decides whether this can run on every turn or only behind an
explicit command. A model that occasionally rewrites someone's config because they said "make the
tests run faster" is worse than no model.

## The floor this has to beat

Before any of it is justified, the same eval set gets run against a **no-neural-network baseline**:
BM25 over the field cards, plus a hand-written rule set for the twenty most common intents
(enable/disable X, set N, change the model, add an MCP server). If that scores within a few points
of the model on the real request distribution, the model is 100 MB of dead weight and the project
should stop at the retriever. This is written down first on purpose - see
[07-roadmap](07-roadmap.md), milestone 0.
# 3. Architecture

The design principle behind every choice below:

> **The model learns intent, not the schema.** Field names, types, defaults and legal values come
> from the schema at runtime. The model's job is to pick from what it was shown and copy it.

Everything that would otherwise force a retrain when Coddy adds a config key is pushed out of the
weights and into a runtime index built from the embedded `config.schema.json`.

## 3.1 The pipeline

Four stages, each of which can fail cheaply into the next-best behaviour.

### Stage 0 - prefilter (no model)

A character-level test that answers "could this possibly be about settings?" with a ~5 µs budget.
Two signals: a small bilingual lexicon (настрой, конфиг, включи, выключи, поставь, лимит, таймаут,
модель, ключ, порт, config, enable, disable, set, limit, timeout, model, token, port ...), and the
presence of any token that prefix-matches a known config path segment. Misses are recovered by
stage 1 when the component runs behind an explicit command; when it runs ambiently on every turn,
the prefilter is what keeps the cost at zero for the 95 % of turns that are about code.

### Stage 1 - gate

Binary decision: config request or not. Two implementations to benchmark against each other:

- **the generator itself**, forced to emit one token (`NOOP` or `EDIT`) as its first output. Free,
  one forward pass over the prompt, no second artefact. Costs a full prefill of the retrieved
  context.
- **a separate tiny classifier** - a 4-8 M parameter bilingual encoder, or even logistic regression
  over character n-grams and retrieval scores. Sub-millisecond, and it can run before retrieval, so
  it saves the retrieval and prefill cost on every non-config turn.

The cheap classifier is the one to build first because it is what makes the ambient mode affordable,
and because its training data is trivially harvestable (every user turn in every Coddy session is a
negative unless it led to a `config_set`).

### Stage 2 - retrieval over field cards

The corpus is generated from the schema, not written by hand. Measured on the real
`config.schema.json` at `df6c258` (see `experiments/schema_index/`):

```
cards total      : 208      (178 addressable, 30 container objects)
with description : 208
with enum        :  13
full corpus      : 35 744 chars, ~9 900 tokens
avg card         : 172 chars, ~48 tokens
```

A card:

```
agent.llm_min_interval_ms : integer  (default=0)
  Minimum gap between consecutive LLM calls in milliseconds, retry attempts included
  (0 disables pacing; e.g. 12000 on strict free tiers).
```

Three consequences:

- the whole schema is ~10 k tokens, which is a 5-20 s CPU prefill for a small model - too slow for
  a fast path, so retrieval is not an optimisation, it is the thing that makes the design possible;
- top-12 cards is ~600 tokens, which with the instruction and the current values of the candidate
  fields lands the prompt at **~800-1000 tokens**, the number the whole latency budget is built on;
- 208 documents is a tiny corpus. This is not a vector-database problem.

Retrieval is hybrid and deliberately boring:

1. **Lexical (BM25)** over path segments, description text and an alias list. Handles "max_turns",
   "telegram", "порт" once the alias file carries the Russian.
2. **Dense** over a static embedding table (model2vec-style: one vector per token, mean-pooled - no
   transformer at query time, a few MB, pure Go, microseconds). Handles "хочу чтобы он не тарахтел
   в консоль" -> `logger.level`.
3. **Section priors** from the current config: fields the operator has already set, and sections
   mentioned in the instruction, get a boost.

The **alias file** is the one generated artefact that needs a big model, and it is generated
offline, once per Coddy release, by asking a frontier model for 5-10 Russian and English paraphrases
per field ("максимальное число ходов", "лимит итераций", "turn cap"). It ships as data next to the
schema. A schema change regenerates it in one cheap batch job - no training involved. If the alias
file is stale or missing, retrieval degrades to descriptions only, it does not break.

### Stage 3 - constrained generation

Input:

```
<task>увеличь лимит ходов до 40 и выключи телеграм</task>
<fields>
1 agent.max_turns : integer (default=12) = 12
    Hard cap on LLM calls per prompt turn.
2 gateways.telegram.enable : boolean (default=false) = true
    Start the Telegram gateway with coddy serve.
...
12 ...
</fields>
```

Output:

```
set agent.max_turns=40
set gateways.telegram.enable=false
```

The decoder runs against a grammar compiled at request time from the retrieved cards. The grammar
is small and regular - the UCI language is four verbs, a path, and a typed value:

```
program   := (line "\n")* | "NOOP"
line      := "set " path "=" value
           | "add_list " path "=" value
           | "del_list " path "=" value
           | "delete " path
path      := one of the candidate paths, as a trie of tokens
value     := bool | int | float | quoted-string | enum-member   (per the card's type)
```

Because the path alternatives are a trie over the *retrieved* paths and the value alternatives come
from the card's declared type and enum, **a syntactically invalid or schema-unknown output is not
representable**. The failure modes that remain are semantic: right syntax, wrong field, wrong
value, missing command, spurious command. Those are what the eval measures and what training has to
fix.

Grammar-constrained decoding is standard now (llama.cpp GBNF, XGrammar, llguidance, outlines); the
open question for this project is what is available from Go, which is the subject of
[02-prior-art](02-prior-art.md) and settled in §3.4.

### Stage 4 - verify and repair

```go
cmds, err := config.ParseUCICommands(lines)   // grammar makes this near-certain
err = config.DryRunUCICommands(paths, cmds)   // applies to a copy, reports the real error
err = config.CheckFile(...)                   // the schema stage of `coddy -t`
```

On failure the error string goes back into the prompt for exactly one retry, then the request is
escalated. On success the commands are staged - the existing `config_set` staging area - and shown
to the operator, who confirms with `config_commit` as they do today.

The verifier is also the training signal. It is a free, exact, non-gameable reward function, which
is why rejection sampling and RL are realistic here and not wishful thinking
(see [05-training](05-training.md)).

## 3.2 Copy, don't recall - and how training enforces it

If the model is fine-tuned on the *current* Coddy schema, it will memorise `agent.max_turns` and
happily emit it whether or not it was retrieved. It will then also emit it for a schema where the
field was renamed. Memorisation is the exact failure this project must not have.

Three mechanisms, applied together:

1. **Schema randomisation during training.** Every training example is generated against a
   *mutated* schema: fields renamed with plausible synonyms, sections shuffled, types and defaults
   changed, fields invented, fields deleted. The model sees the real Coddy schema in only a minority
   of examples. This is the same move that took text-to-SQL from single-database systems to
   cross-database generalisation on Spider, and the failure it prevents is identical.
2. **Index-anchored supervision.** Targets are written so the correct answer is only derivable from
   the candidate list, never from world knowledge - including negative examples where the plausible
   field is deliberately absent from the candidates and the correct answer is an abstention.
3. **Grammar at inference.** Even a memorising model cannot emit a path outside the trie.

The natural extension of (2) is to make the model emit the *card index* (`set #1=40`) instead of the
path, which shortens the output, removes path tokenisation entirely, and makes copying structural.
The reason to keep literal paths as the default is debuggability: a log line that reads
`set agent.max_turns=40` is readable by the operator and by the existing tooling, and the grammar
already gives the same guarantee. Index output stays in reserve for the smallest model tier, where
every output token counts.

## 3.3 Which model

Two tracks, deliberately sequenced so the risky one is optional.

**Track A - conventional small model, quantised.** Fine-tune a 270 M-600 M base with a permissive
licence, quantise to int4/int8 GGUF, ship 150-400 MB. Known-good tooling end to end. The point of
track A is not the artefact - it is that it produces the data pipeline, the grammar, the Go
integration and the eval harness, all of which track B needs verbatim.

**Track B - ternary (BitNet b1.58-style).** The article that started this project
([notes/habr-1074678.md](../notes/habr-1074678.md)) is a hobby-scale demonstration of the recipe:
BF16 master weights, per-layer mean-absolute-value threshold, straight-through estimator, ternary
weights materialised on every forward pass. The reason to care here is not the hype, it is one
specific engineering fact:

> A ternary matmul is additions and subtractions. A Q4_K matmul is not.

Which means ternary is the one low-bit format whose kernels are realistic to write **in pure Go**,
with no cgo, for a binary that already cross-compiles to six platform/arch pairs and refuses to
carry a C toolchain in the default build. Ternary also puts a 300 M model at roughly 60-80 MB, which
is `go:embed`-able, where a Q4 model of the same capability is not.

The cost is that ternary weights cannot be produced by post-training quantisation of a normal
checkpoint at acceptable quality - they need quantisation-aware training or distillation. That is a
real GPU bill, quantified in [05-training](05-training.md), and it is the reason track B comes
second.

## 3.4 Living inside the Coddy binary

The integration surface is intentionally thin.

```
external/tinyconfig/          // build tag `tinyconfig`, mirrors external/{cli,ui,swarm,...}
    cards.go                  // build field cards from the embedded schema
    retrieve.go               // BM25 + static embeddings, no dependencies
    grammar.go                // trie -> decoding constraint
    infer_*.go                // the runtime, behind sub-tags
    bridge.go                 // Suggest(ctx, instruction, cfgPath) ([]string, Confidence, error)
internal/tools/config_set.go  // unchanged
```

The untagged core sees an interface with a stub implementation, the way `internal/serve` already
treats every optional surface (`Available` consts and stubs). Default `make build` links none of it.

Open decisions, to be closed in [07-roadmap](07-roadmap.md) once the runtime survey lands:

- **weights in the binary or beside it.** `go:embed` of 60-80 MB (ternary) is defensible and makes
  the feature work offline with zero setup; 400 MB (track A) is not - that path downloads into
  `~/.coddy/models/` on first use, with the same checksum-and-resume machinery `coddy update`
  already has. The packaging rules in `CLAUDE.md` matter here: the release archive layout is
  referenced by the Homebrew cask, so an extra artefact is not free.
- **cgo or not.** A cgo-linked llama.cpp is fast and mature and turns the release matrix into a
  cross-compilation project. Pure Go is slower and is work we own. The staged answer: track A behind
  a cgo sub-tag for developers, track B pure Go for the shipped default.
- **tokenizer.** Must be pure Go and byte-exact with the training tokenizer, or every guarantee
  above evaporates.

## 3.5 Latency budget

At 4 threads on a mid-range laptop CPU, per request:

| Stage | Budget | Note |
|---|---|---|
| prefilter | < 0.1 ms | pure string work |
| gate classifier | < 5 ms | tiny encoder or logistic regression |
| retrieval | < 10 ms | 208 documents |
| prefill ~900 tokens | 300-800 ms | the dominant cost, and why retrieval exists |
| decode ~30 tokens | 100-400 ms | short outputs by construction |
| verify | < 5 ms | parse + dry-run on a copy |

The ambient mode (run on every turn) only pays the first two rows, ~5 ms, for turns that are not
about configuration. That is the whole reason the gate is a separate model.
# 4. Data

There is no dataset for this task and there will not be one donated. The plan is built around that:
the bulk of training data is generated, its correctness is machine-checked rather than trusted, and
the scarce real data is spent on evaluation instead of training.

## 4.1 What real data exists today

Coddy sessions persist as `~/.coddy/sessions/<id>/messages.json`, complete with tool calls, so the
pairs we want are in principle already on disk. Measured on this machine: 111 sessions, 10 of which
so much as mention `config_set`, and most of those mentions are in reasoning text rather than in an
actual tool call.

That is the honest baseline, and it says two things. Real logs are a **gold evaluation set and a
seed for the intent taxonomy**, not a training corpus. And the **negative class is abundant** - every
user turn that did not lead to a config edit is a labelled no-op, and there are tens of thousands of
those across a modest user base.

## 4.2 The generator

The pipeline runs backwards, because the forward direction (instruction -> commands) is the hard one
and the backward direction (commands -> instruction) is the one a teacher model does reliably. This
is instruction backtranslation, and it is the standard move for structured targets: generate the
verified artefact first, then write the request that would have produced it.

```
 1. sample a schema        real schema, or a mutation of it (§4.3)
 2. sample a config        plausible config.yaml consistent with that schema
 3. sample an edit         1-3 UCI commands, drawn from the intent taxonomy (§4.4)
 4. apply + verify         ParseUCICommands -> DryRunUCICommands -> schema check
                           reject silently if any stage fails; this is free and exact
 5. write the request      teacher LLM: "an operator wants exactly this change - what would they
                           type?" in Russian and in English, 2-4 paraphrases, varying register
 6. round-trip check       a second, independent teacher sees only (config, request, candidate
                           cards) and must reproduce the commands; disagreement means the request
                           was ambiguous -> it goes to the abstention pool, not the positive pool
 7. build the context      run the real retriever to get candidate cards, so training sees exactly
                           the imperfect top-k it will see at inference, misses included
```

Step 4 is the part that makes this cheap. Coddy's own `internal/config` is the verifier, so a
generated example is either provably applicable or discarded, and no human ever reviews a target.
Step 6 is the part that makes it honest: an example whose request does not determine the edit is a
training example for *abstention*, which is where most of the safety of the final system comes from.

Step 7 matters more than it looks. If training always shows the gold field in the candidate list,
the model learns that the answer is always present and will confabulate when retrieval misses. A
fixed fraction of examples (target: 15-20 %) must have the gold field **absent** from the candidates,
with abstention as the correct answer.

## 4.3 Schema mutation

Every example is generated against a schema drawn from:

| Variant | Share | Purpose |
|---|---|---|
| the real Coddy schema | 25 % | the deployment target |
| renamed fields (`max_turns` -> `turn_limit`, `iteration_cap`) | 25 % | kill memorisation of names |
| restructured sections (moved, merged, split, renamed) | 20 % | kill memorisation of paths |
| type and default changes (bool -> enum, int range moved) | 15 % | force reading the card, not recalling |
| invented sections and fields | 15 % | fields that will exist in future releases |

The rename set is generated once by a teacher model and cached; the rest is mechanical. The
evaluation set adds a variant the training set never sees at all: **the schema of a later Coddy
release**, held out precisely to measure the property the whole design is built on.

The reason to be this deliberate is [SGD-X](https://arxiv.org/abs/2110.06800): in schema-guided
dialogue, 71 % of "unseen" intent names and 65 % of "unseen" slot names turned out to appear in
training, so the published zero-shot numbers were inflated. The same trap is waiting here, and
mutation plus a genuinely later schema is how to avoid walking into it.

## 4.4 Intent taxonomy

Drawn from the real command surface, weighted by what operators actually ask. Each row generates
both positive examples and near-miss negatives.

1. toggle a boolean (`httpserver.enable`, `gateways.telegram.enable`);
2. set a number, absolute ("до 40") and relative ("в два раза больше", which needs the current value);
3. switch the default model or a per-model setting;
4. add or remove a list entry (`add_list skills.dirs=...`, `del_list`);
5. edit a selector-addressed entry (`mcp_servers[name=context7].command`);
6. change a path or directory;
7. change a timeout, retry or interval, with unit conversion ("полминуты" -> 30000);
8. secrets and credentials (`providers[name=x].api_key`), where the correct behaviour is usually to
   stage a `${ENV}` reference rather than a literal;
9. multi-field requests that need 2-4 commands;
10. delete or reset a field back to its default;
11. **no-op: not about configuration** (the majority class in ambient mode);
12. **no-op: already in that state** (requires reading the current value);
13. **abstain: ambiguous** ("сделай побыстрее" - which of eight timeouts?);
14. **abstain: not expressible** ("сделай так, чтобы он не тупил").

Rows 11-14 are not an afterthought. In ambient deployment they are 95 % of traffic, and the ship
criteria in [01-problem](01-problem.md) are dominated by them.

## 4.5 Linguistic coverage

Russian and English in roughly equal share, with the Russian half carrying the harder variation
because that is where a small model degrades first: imperative and infinitive forms, colloquial verbs
(врубить, вырубить, скрутить, поднять), transliterated technical nouns (таймаут, воркер, гейтвей),
mixed-script requests ("поставь agent.max_turns в 40"), typos and missing diacritics, and the habit
of naming a field by its UI label rather than its path. English gets terser, more imperative forms
and more direct path references. A small share (5 %) is deliberately mixed-language in one sentence,
because that is how bilingual operators actually type.

## 4.6 Volume

| Stage | Examples | What it buys |
|---|---|---|
| smoke | 1 000 | first signal; the Kubernetes-manifest SLM study reached 91 % full-pass at exactly this size with a 1.5B model, so 1k is not a token gesture |
| v1 | 20 000-50 000 | production candidate for a 270M-600M model |
| v2 | 100 000-200 000 | schema-mutation coverage deep enough to trust the unseen-field claim |
| negatives | 50 000+ | harvested and synthesised; cheap, and the gate lives on them |

Teacher cost at v1, generating 4 paraphrases per verified edit with a mid-tier model: on the order of
10-20 M output tokens, which is tens of dollars, not thousands. The round-trip check doubles it.

## 4.7 Donated real data

An opt-in path, built as a normal Coddy feature rather than telemetry:

- `coddy dataset export` walks the local sessions, finds turns that led to a committed config change,
  and writes (instruction, config skeleton, commands) triples **to a local file**;
- redaction is mandatory and runs before anything is written: `RedactedString` already exists in
  `internal/config/uci.go` for exactly this, api keys and tokens never leave the machine, and paths
  under `$HOME` are rewritten;
- the operator reviews the file and sends it themselves. No background upload, no opt-out telemetry,
  no "anonymous usage statistics" checkbox that defaults to on;
- what comes back is licensed permissively and published, so the dataset outlives the project.

Expected yield is small - the measurement above says single-digit examples per active user per
month. It is still worth building, because those examples are the only ones drawn from the real
request distribution, and the eval set needs exactly that.

## 4.8 Contamination control

- eval instructions are generated by a different teacher model than training instructions;
- the eval set includes a schema from a Coddy release later than the training snapshot;
- donated real examples are eval-only, always;
- exact-duplicate and near-duplicate (normalised edit distance) filtering across the train/eval
  boundary, run as a CI check on the dataset rather than as a one-off script.
# 5. Training and what the hardware costs

The headline, up front, because it changes how the project should be planned:

> **The supervised half of this project costs single-digit dollars of rented GPU time.** The
> expensive parts are generating the synthetic data and, if track B goes ahead, the ternary
> conversion. Nothing here needs a cluster.

## 5.1 Base model

| Candidate | Params | Licence | Verdict |
|---|---|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6 B (0.44 B non-embedding), vocab 151 936 | Apache 2.0 | **primary.** Clean licence, real multilingual pretraining, the only sub-1B with credible Russian |
| [Qwen3-1.7B](https://qwenlm.github.io/blog/qwen3/) | 1.7 B | Apache 2.0 | fallback if 0.6 B underfits; too big to embed, fine to download |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360 M | Apache 2.0 | the small end. Fully open recipe, but weak Russian |
| [Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/) | 270 M | Gemma Terms of Use | technically ideal (built for exactly this), legally awkward: the use restrictions must flow down to every downstream user, and Google reserves unilateral termination. Bad fit for weights bundled in an OSS CLI |
| [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) | 350 M | LFM Open License v1.0 | free only under $10 M revenue - disqualified for redistribution |
| [Llama-3.2-1B](https://www.llama.com/llama3_2/license/) | 1.2 B | Llama Community | naming and MAU conditions; avoidable, so avoid |

Licence is a hard filter here, not a preference: the artefact ships inside a binary other people
redistribute.

Two Russian-specific options worth measuring rather than assuming:
[Vikhr-Qwen-2.5-0.5B-Instruct](https://huggingface.co/Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct)
(SFT on 150 k Russian instructions) and
[RuadaptQwen2.5-1.5B](https://huggingface.co/RefalMachine/RuadaptQwen2.5-1.5B-instruct), which
re-trains the tokenizer for Russian and therefore cuts the token count of every Russian prompt.
Tokenizer efficiency matters more than usual here because the prompt is the latency budget.

## 5.2 The recipe

Four stages, each with a stopping rule.

1. **SFT on verified synthetic data.** Loss on the completion only. Packing off (examples are short
   and the prompt structure is the signal). Stop when exact-match on the dev set plateaus.
2. **Rejection-sampling fine-tuning.** Sample k=8 completions per training prompt from the SFT
   model, keep the ones the verifier accepts and that reproduce the gold post-edit config, retrain
   on them. This is the cheapest real gain available, because the verifier is free and exact.
3. **GRPO with a verifiable reward.** Reward = parses (1.0) + dry-runs clean (1.0) + schema-valid
   (2.0) + resulting config equals gold (4.0) - spurious command (-2.0) - wrong no-op (-4.0).
   Expect single-digit to low-teens points, concentrated exactly where this project is fragile: no-op
   precision and multi-command requests.
4. **Distillation to ternary (track B only).** See §5.5.

Evidence for the shape of this: a 350 M model fine-tuned on 5 000 synthetic examples went from
**61.4 % to 98.0 %** shell-command tool-call accuracy, beating its own 120 B teacher
([distil labs](https://www.distillabs.ai/blog/fine-tuning-liquids-lfm25-accurate-tool-calling-at-350m-parameters/));
a Qwen2.5-Coder-1.5B fine-tuned for Kubernetes manifests hit **91 % full-pass** (syntax + strict
schema + semantics + security) with **1 000 training examples**
([paper](https://arxiv.org/html/2605.25835v1)); and Qwen2.5-1.5B trained with GRPO for structured
output beat DeepSeek-R1-671B on the same structured metric
([ThinkJSON](https://arxiv.org/html/2502.14905v1)). Also settled in the literature: **RL alone does
not establish the output format** - SFT initialisation is required, so stage 3 never replaces
stage 1.

## 5.3 Workload arithmetic

Our sequences are short: a retrieved-card prompt is ~900-1 200 tokens and the completion is ~30.
Call it 1 200 tokens per example.

```
FLOPs  = 6 · N · D                      (forward + backward, dense)
D      = examples × tokens × epochs
```

| Job | N | D | FLOPs |
|---|---|---|---|
| SFT v1 | 0.6 B | 50 k × 1.2 k × 3 = 180 M | 6 × 6e8 × 1.8e8 = **6.5e17** |
| SFT v2 | 0.6 B | 200 k × 1.2 k × 3 = 720 M | **2.6e18** |
| SFT v1 on 360 M | 0.36 B | 180 M | **3.9e17** |
| Rejection sampling (generation, k=8 over 50 k) | - | ~400 M output tokens at 2·N·D | ~4.8e17 |
| GRPO, 1.5 k steps | 0.6 B | ~ | ~2e18 equivalent |

Effective throughput, dense BF16 at MFU 0.35-0.45: RTX 3090 ≈ 25 TFLOPS, RTX 4090 ≈ 65,
A100-80 ≈ 120, H100-SXM ≈ 380. (Sanity check: a hand-written CUDA pipeline reaches 39 k tok/s BF16
on a 0.5 B model on one 4090, i.e. 117 TFLOPS at 73 % MFU -
[LLMQ](https://arxiv.org/html/2512.15306v1) - so 65 for a PyTorch + Flash-Attention pipeline is
deliberately conservative.)

| Job | 1× 3090 | 1× 4090 | 1× A100-80 | 1× H100 | $ at 4090 $0.34/h |
|---|---|---|---|---|---|
| SFT v1 (0.6 B, 50 k ex.) | 7.2 h | 2.8 h | 1.5 h | 0.5 h | **$0.95** |
| SFT v2 (0.6 B, 200 k ex.) | 29 h | 11 h | 6 h | 1.9 h | **$3.80** |
| Rejection-sampling pass | ~5 h | ~2 h | 1.1 h | 0.4 h | $0.70 |
| GRPO 1.5 k steps | 40 h+ | 8-24 h | 6 h | 2-4 h | ~$5 |
| **Whole SFT+RS+GRPO programme** | - | **~1.5 days** | - | - | **under $15** |

Rental rates sampled 2026-08-11: RunPod community 4090 $0.34/h, A100-80 $1.19/h, H100 SXM $2.69/h;
Vast.ai 4090 $0.34-0.50, verified H100 $1.50-1.87.

### VRAM

Full fine-tuning, AdamW mixed precision, is 16 bytes/param (BF16 weights 2 + BF16 grads 2 + FP32
master 4 + m 4 + v 4): **9.6 GB for 0.6 B**, 5.8 GB for 360 M. That fits a single 24 GB card with
room to spare, which is why **LoRA is not needed below 1 B** - and full fine-tuning generalises
better under the distribution shift that schema mutation deliberately creates.

The trap is not the optimiser, it is the vocabulary. At 151 936 tokens, the logits tensor for one
1 200-token sequence is 1 200 × 151 936 × 4 B = **730 MB in FP32**, and 2-3 GB once gradients and
softmax intermediates are counted. Gemma 3 270M is worse (256 k vocab). Micro-batch stays at 2-4 on
a 24 GB card unless fused linear cross-entropy (Liger-Kernel, Cut Cross-Entropy) is used - with it,
micro-batch 16 is reachable. Budget a day for this detail rather than discovering it at 3 a.m.

## 5.4 The real cost centre: synthetic data

Generating 50 k verified examples with 2-4 bilingual paraphrases each, plus a round-trip check by a
second model, is on the order of 20-40 M teacher output tokens. At commercial mid-tier API prices
that is **$150-400**; on the operator's own inference hub it is electricity and a weekend. Given
that Coddy already talks to a self-hosted hub, generating the corpus locally is the obvious move,
and it also removes any question about teacher-output licensing in the published dataset.

Reusable generators worth not rewriting:
[distilabel](https://github.com/argilla-io/distilabel) (typed, composable steps - the best fit for a
deterministic schema-driven pipeline), [Bespoke Curator](https://github.com/bespokelabsai/curator)
(has code-execution backends, so the YAML verifier can run inline). The naming for what §4.2 does is
established: **instruction backtranslation** ([Humpback](https://arxiv.org/abs/2308.06259), reverse
instructions in [LongForm](https://arxiv.org/abs/2304.08460)), with
[constraint back-translation](https://arxiv.org/pdf/2410.24175) as the closest analogue.

## 5.5 Track B: the ternary conversion

Two ways to get a ternary model, and only one of them is sane.

**From scratch.** The [Spectra/TriLM](https://arxiv.org/abs/2407.12327) regime: ternary models
trained from initialisation, 99 M-3.9 B params on 300 B tokens. A 200 M ternary model on 5 B tokens
is 6 × 2e8 × 5e9 × 1.3 (QAT overhead) = 7.8e18 FLOPs ≈ **33 h on one 4090, ~$11**. Affordable and
useless: at 25 tokens per parameter the model will have no Russian and barely any English, where
Qwen3-0.6B saw thousands of tokens per parameter. Taking it to 100 B tokens costs ~$205 and a week
of 8×H100, and still lands below a fine-tuned 0.6 B.

**Distillation from the model we already trained.** [BitNet Distillation](https://arxiv.org/abs/2510.13998)
(Microsoft, Oct 2025) is the recipe: SubLN surgery on the FP16 checkpoint, ~10 B tokens of continual
pretraining to let the weights settle into the ternary basin, then dual distillation (logits +
attention relations) from the FP16 teacher, reporting near-FP16 quality with up to 10× memory saving
and 2.65× faster CPU inference.

```
0.6 B × 10 B tokens × 1.3 QAT overhead = 6 × 6e8 × 1e10 × 1.3 = 4.7e19 FLOPs
   1× 4090  : 200 h  (8.3 days)   ≈ $68
   8× H100  : 4.8 h               ≈ $60-105 depending on provider
```

So: **the ternary version costs about a hundred dollars and a week of one desktop GPU**, on top of a
track A model that already works. That is the number that decides track B, and it is small. What it
does not buy is certainty - the published distillation results are for models at and above 0.6 B on
general benchmarks, not for a 300 M model on a narrow bilingual structured task, and nobody has
published that data point. It is a genuine experiment, which is why it is sequenced after a working
track A rather than instead of one.

The other half of track B's risk is not training at all, it is inference: see
[02-prior-art](02-prior-art.md) §Go runtimes. A ternary model is only worth having if something can
run it from Go.

## 5.6 What can be skipped

- **LoRA/QLoRA.** Not needed under 1 B; use it only if the experiment grid grows to dozens of runs
  where adapter swapping is convenient.
- **RL before SFT.** Does not work, repeatedly shown.
- **A bigger base "to be safe".** The failure mode at this size is data coverage, not capacity, and
  a 1.7 B model cannot be embedded in the binary at any quantisation.
# 6. Evaluation

The eval harness is built before the model, because the decision this project has to keep making -
is the model earning its bytes? - is only answerable against a fixed ladder of baselines.

## 6.1 The ladder

Every number gets reported for all four rungs, on the same sets:

| Rung | What it is | Why |
|---|---|---|
| R0 | BM25 over field cards + ~20 hand-written intent rules, no neural net | the floor. If the model cannot clear this by a wide margin, the project is a retriever |
| R1 | retrieval + grammar + a **frontier model** | the ceiling. Anything R1 cannot do is a data or task-definition problem, not a model-size problem |
| R2 | retrieval + grammar + our fine-tuned small model | the deliverable |
| R3 | R2 with the ternary weights | track B, only if it lands within a few points of R2 |

R1 matters more than it looks: if a frontier model with the same retrieval and grammar scores 0.72
exact-match, then the retrieval or the task definition is the bottleneck and no amount of fine-tuning
a 600 M model will fix it.

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

**No-op confusion matrix**, reported separately for Russian and English, and separately for the two
no-op classes (not about config / already in that state). Precision here is the ship gate.

**Retrieval recall@k** (is the gold field in the top k) at k = 5, 12, 25, borrowed straight from the
text-to-SQL schema-linking literature. Retrieval recall is a hard ceiling on everything downstream:
the generator cannot emit what it was never shown.

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

**Kill criterion:** if R0 (rules + retrieval) lands within ~10 points of R1 (frontier model with the
same retrieval) on the real request distribution, ship R0 as a Coddy feature and stop. A 100 MB model
that replicates a regex table is a liability.

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

**Kill criterion:** if no cgo-free Go path meets the latency budget, the feature ships behind a cgo
sub-tag for the people who want it and is not in the default build. That is a smaller, still-useful
product - not a failure.

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
60-80 MB is embeddable in the binary, 400 MB is not - and that only matters if the model is good
enough to be worth embedding.

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
