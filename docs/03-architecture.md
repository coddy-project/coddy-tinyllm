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
schema. A schema change regenerates it in one cheap batch job - no training involved.

This is where the "no retraining" promise actually lives, and it is worth being precise about it.
The model never needs retraining; **the alias file does need regenerating**, because retrieval recall
on a brand-new field is what caps everything downstream - a perfect generator with 0.85 recall on new
fields scores 0.85. So regeneration is a **release-blocking step**, in the same category as a schema
migration, not an offline batch job that degrades gracefully. A stale alias file does not break the
feature loudly; it quietly moves it below its own floor, which is worse.

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
from the card's declared type and enum, **a syntactically malformed command and an out-of-candidate
path are not representable**. That is the whole of the guarantee, and it is worth stating narrowly:
a grammar of this shape does *not* enforce numeric ranges, string patterns, format constraints,
nullability, union members, cross-field invariants or list-element structure. Those stay with the
dry-run and the schema check in stage 4, which is why stage 4 is not optional.

Two mechanical details the guarantee depends on:

- **selector expansion.** `mcp_servers[name=context7].command` is not a schema path - the selector
  value comes from the *current config*. The path trie is therefore built from the schema's static
  paths **instantiated against the live document**: every list entry contributes its concrete
  selector, so the model chooses among servers that exist rather than inventing one. Creating a new
  entry is a separate, explicitly enumerated alternative.
- **schema-construct conformance.** The compiler from schema to grammar needs its own test suite
  covering every construct the real schema uses - `oneOf`, nullable types, arrays of objects,
  enums, patterned strings - before "validity by construction" may be claimed anywhere.

The failure modes that remain are semantic: right syntax, wrong field, wrong value, missing command,
spurious command. Those are what the eval measures and what training has to fix.

### The five terminal states

`NOOP` is not one outcome, and collapsing it into one token loses exactly the distinctions the
caller needs:

| State | What the caller does |
|---|---|
| `EDIT` | stage the commands as a proposal |
| `NOT_CONFIG` | return control to the agent; the turn was never ours |
| `ALREADY_SATISFIED` | tell the operator the setting is already that way, change nothing |
| `AMBIGUOUS` | ask one clarifying question, or escalate |
| `UNSUPPORTED` | escalate to the full agent - red-class field (§3.6), or beyond the DSL |

`ALREADY_SATISFIED` cannot be decided by the gate, because it needs the current values, so it lives
after retrieval. `UNSUPPORTED` is partly decided in Go, before the model runs, by the writable-surface
policy. Confidence is derived outside the model - from retrieval scores, margin between candidates
and an out-of-distribution signal - rather than trusted from a self-reported token, because a small
model's stated confidence is not calibrated.

Grammar-constrained decoding is standard now (llama.cpp GBNF, XGrammar, llguidance, outlines); the
open question for this project is what is available from Go, which is the subject of
[02-prior-art](02-prior-art.md) and settled in §3.4.

### Stage 4 - verify and repair

```go
cmds, err := config.ParseUCICommands(lines)   // grammar makes this near-certain
err = config.DryRunUCICommands(paths, cmds)   // applies to a copy, reports the real error
err = config.CheckFile(...)                   // the schema stage of `coddy -t`
```

Three checks that belong here rather than in the model, because they are deterministic and free:

- **already satisfied.** After the model emits `set gateways.telegram.enable=false`, compare against
  the current value. If the document would not change, the answer is `ALREADY_SATISFIED`, whatever
  the model thought it was doing. This removes the hardest abstention class from the model's
  responsibility entirely.
- **semantic plausibility.** A cheap surprise score over (instruction, command): does a number in the
  instruction match the value being set; do the field's path segments, description or aliases share
  tokens with the instruction; is the value at an extreme of the field's range. A high-surprise edit
  is not rejected - it is escalated to explicit confirmation even in ambient mode. This is the
  guard against "make the tests run faster" turning into `set agent.max_turns=4`, which passes every
  other check in this list.
- **command count.** More than five commands from one instruction is an abstention, not an answer.
  The grammar bounds it, the training data bounds it, and the point is the operator's review burden:
  a diff nobody reads is not a safety mechanism.

On failure the error string goes back into the prompt for exactly one retry, then the request is
escalated. On success the commands become a **proposal**, and this is where the design has to be
careful.

Coddy's staging area is per-session and persistent: `internal/tools/config_staging.go` writes
`config_staging.json` into the session directory so pending commands survive restarts, and a later
`config_commit` commits *everything* staged. Writing an unasked-for ambient suggestion into that area
would mean a stray edit can ride along with a legitimate commit the operator makes twenty minutes
later. So:

- an **explicit** request (`coddy config "..."`) may stage normally - the operator asked, and the
  next thing they see is the diff;
- an **ambient** suggestion never touches the shared staging area. It produces an isolated proposal
  carrying the source turn, a hash of the config it was computed against, the exact diff and an
  expiry, and it is staged only after the operator accepts it - and rejected if the config changed in
  the meantime (compare-and-swap on the hash).

"Shown to the operator" is a weak control on its own, because confirmation fatigue is real. The
isolation is what makes the failure mode recoverable.

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
carry a C toolchain in the default build.

### What bit-width this actually is

"1.58-bit" describes a weight's information content, `log2(3)`. Nothing stores it at 1.58 bits, and
nothing stores the whole model at any single width. Three separate numbers get conflated:

| | bits/weight | what it is |
|---|---|---|
| information in a trit | 1.585 | the theoretical floor |
| `TQ1_0`, `TQ2_0` (llama.cpp) | 1.6875, 2.0625 | how the **linear layers** are packed, scales included |
| `I2_S` (bitnet.cpp) | 2.0 | the same, in Microsoft's format |
| `Q4_K_M` | ~4.85 | track A, everything |

And the linear layers are not the whole model. The **embedding and output head stay 4-8 bit** in every
BitNet recipe, and in a small model with a large multilingual vocabulary they are a quarter of the
parameters. Computed from the real configs (`experiments/bitness.py`):

| model | embedding share | all Q4_K_M | TQ2_0 + Q8 embed | TQ2_0 + Q4 embed | TQ1_0 + Q4 embed |
|---|---|---|---|---|---|
| Qwen3-0.6B (vocab 151 936) | 26 % | 345 MB / 4.85 bpw | 266 MB / **3.74** | 192 MB / **2.70** | 172 MB / **2.42** |
| SmolLM2-360M (vocab 49 152) | 13 % | 209 MB / 4.85 | 125 MB / **2.90** | 103 MB / **2.38** | 89 MB / **2.05** |
| SmolLM2-135M | 21 % | 78 MB / 4.85 | 55 MB / **3.42** | 41 MB / **2.58** | 37 MB / **2.28** |
| Gemma 3 270M (vocab 262 144) | 63 % | 156 MB / 4.85 | - | 116 MB / **3.60** | - |

So the honest description of a "ternary" artefact here is **1.58-bit linear layers inside a
2.4-3.7 bit-per-weight file**, and the real memory ratio against Q4_K_M is **1.3-2.4×**, not 4× and
not even the 2.2-2.8× the packing formats alone suggest. Gemma 3 270M is the cautionary case: 63 % of
its parameters are a 256 k-token embedding table that ternarisation does not touch, so converting it
buys almost nothing.

Two consequences for this project. **Vocabulary size matters as much as parameter count** - trimming
Qwen3-0.6B's vocabulary to the ~32 k tokens this task actually needs takes the ternary artefact to
142 MB, more than the jump from Q4 to ternary does. And **"ternary" and "embedded in the binary" only
coincide at 360 M or below with a small vocabulary** - which is also where the Spectra scaling law
bites hardest. That tension is the real content of track B, and it is resolved by measurement, not by
preference.

The cost is that ternary weights cannot be produced by post-training quantisation of a normal
checkpoint at acceptable quality - they need quantisation-aware training or distillation. That is a
real GPU bill, quantified in [05-training](05-training.md), and it is the reason track B comes
second.

The pure-Go-kernels argument above turns out to be moot in the good direction: `TQ1_0` and `TQ2_0`
have been in mainline ggml since 2024 and the Go runtime carries them, so a ternary model rides the
same engine as a Q4 one and nobody writes ternary kernels in Go ([02-prior-art](02-prior-art.md) §H).
What ternary still buys is **size** - roughly 90-190 MB against 210-345 MB depending on the base and
how the embedding is quantised (§3.3) - and about 1.85x decode speed. What it no
longer needs to justify is an inference research project.

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

- **weights in the binary or beside it.** `go:embed` of ~90-140 MB (a ternary 360 M model, or a
  vocabulary-trimmed 0.6 B - §3.3) is arguable and makes the feature work offline with zero setup;
  345 MB (track A on Qwen3-0.6B) is not - that path downloads into
  `~/.coddy/models/` on first use, with the same checksum-and-resume machinery `coddy update`
  already has. The packaging rules in `CLAUDE.md` matter here: the release archive layout is
  referenced by the Homebrew cask, so an extra artefact is not free.
- **the runtime.** [goccy/go-llama](https://github.com/goccy/go-llama) (MIT) is llama.cpp
  transpiled from wasm to Go source, so `CGO_ENABLED=0` holds and all six targets cross-compile from
  one machine; it adds ~20 MB to the binary, exposes GBNF through `Params.Grammar`, and its kernel
  index covers the k-quants **and both ternary types** (`tq1_0`, `tq2_0`), so track A and track B
  share one runtime. Two conditions: the tagged build must set **`GOAMD64=v2`** (at v1 the engine
  falls back to scalar Go, ~1000x slower by the project's own measurement), and the dependency is
  three months old, so M3 in [07-roadmap](07-roadmap.md) is a spike with a stated fallback.
- **tokenizer.** Must be pure Go and byte-exact with the training tokenizer, or every guarantee
  above evaporates.

## 3.5 Latency budget

At 4 threads on a mid-range laptop CPU, per request:

| Stage | Budget | Note |
|---|---|---|
| prefilter | < 0.1 ms | pure string work |
| gate classifier | < 5 ms | tiny encoder or logistic regression |
| retrieval | < 10 ms | 208 documents |
| **prefill ~900-1200 tokens** | **the whole problem, see below** | |
| decode ~30 tokens | 100-400 ms | short outputs by construction |
| verify | < 5 ms | parse + dry-run on a copy |

The ambient mode (run on every turn) only pays the first two rows, ~5 ms, for turns that are not
about configuration. That is the whole reason the gate is a separate model.

### Prefill is the threat to this design, not decode

Worth being blunt, because the first draft of this table was optimistic to the point of being wrong.
Our workload is **prompt-heavy and output-light** - roughly 1 000 tokens in, 30 tokens out - which is
the opposite of the chat workload every published tokens/second figure describes. Those figures are
*generation* speed; ours is dominated by *prompt processing*.

Prefill costs about `2·N·T` FLOPs - for a 0.6 B model and 1 000 tokens, ~8.8e11 - and four AVX2
threads sustain something in the region of 50-150 GFLOPS on quantised GEMM. That is **seconds, not
hundreds of milliseconds**, and on the older laptop this repository was written on it is worse. No
published measurement for this exact combination was found, so the honest statement is: the prefill
cost of the retrieved-card prompt is **unmeasured, plausibly 1-5 s, and is the single biggest threat
to the premise of the project.**

Which is why **M3 measures it before anything is trained** ([07-roadmap](07-roadmap.md)), on stock
weights through the intended runtime, on all three reference machines. The mitigations, in the order
they should be reached for:

1. **fewer and shorter cards** - top-5 instead of top-12, descriptions trimmed to one clause for
   all but the top-3 candidates. The prompt is the budget, so this is the biggest lever;
2. **a smaller generator** - 270-360 M roughly halves prefill against 0.6 B;
3. **prefix reuse** - the instruction block is the only part that changes between the gate pass and
   a repair retry; the KV cache for the card block can be reused across the retry;
4. **do not generate at all for the common case** - see §3.7.

If the measurement lands at the pessimistic end, item 4 is not a fallback, it is the design.

## 3.6 The writable surface, and why it is not the whole schema

A config field is not just data. Measured against the real schema (`experiments/schema_index/`),
39 of 208 fields are security-relevant, and a handful are straightforwardly code execution:

| Field | What writing it does |
|---|---|
| `mcp_servers[].command`, `.args` | starts an arbitrary process |
| `providers[].api_key_command` | runs a shell command as a credential helper |
| `hooks.files`, `hooks.enable`, `hooks.project_trust` | registers and enables lifecycle scripts |
| `tools.command_allowlist` | widens what the agent may run without asking |
| `skills.dirs`, `subagents.dirs` | loads definitions from a new path |
| `providers[].api_base`, `.proxy` | redirects every LLM request, credentials included |
| `*.token`, `*.api_key`, `httpserver.login.password_hash`, `swarm.pairing_tokens` | credentials |

A 600 M model has no security judgement, and the instruction that reaches it is text. So the fast
path gets an **explicit writable-surface policy**, enforced in Go, outside the model:

- **green** - scalars with bounded effect (limits, timeouts, levels, booleans, model ids): the small
  model may stage them;
- **amber** - paths, URLs, directories: staged, but the diff is rendered with a warning and the
  commit needs an explicit confirmation rather than a default-yes;
- **red** - the execution and credential fields above: **the small model may not emit them at all.**
  The grammar simply does not contain those paths. A request that needs one is an escalation to the
  full agent, where the existing permission machinery applies.

Two related rules, for the same reason:

- the fast path reads **only the operator's own typed instruction**. Never tool output, never file
  contents, never a fetched page, never an `AGENTS.md`. Text from those places is exactly the channel
  a prompt injection arrives through, and a component whose whole job is to turn text into config
  writes must not be reachable from it;
- red-class fields stay red even when the operator asks explicitly. The answer is "run that through
  the agent", which costs one round trip and keeps one code path for anything dangerous.

This costs coverage - roughly a fifth of the schema - and it is the difference between a convenience
feature and a remote-code-execution surface with a language model in the middle.

## 3.7 The alternative that might be better than generation

Generation is the obvious shape for this task, not necessarily the right one. Strip the problem down
and it is three decisions, only one of which needs a language model to *write* anything:

1. is this about configuration? - **binary classification**
2. which field? - **ranking over the retrieved candidates**
3. what value? - **span extraction plus normalisation** ("до 40" -> 40, "полминуты" -> 30000,
   "выключи" -> false)

All three are encoder-shaped. A cross-encoder reranker at 20-50 M parameters scores
(instruction, card) pairs; a small extractive head reads the value off the instruction; the verb
follows from the value and the field type (`delete` when the request is "reset to default",
`add_list` when the field is an array and the request is additive). A 30 M encoder over a
1 000-token input is ~6e10 FLOPs - **two orders of magnitude cheaper than a 0.6 B decoder's
prefill**, which is the difference between 200 ms and several seconds.

What it cannot do: multi-command requests that need a plan, values that must be composed rather than
extracted (a URL assembled from parts), and the graceful "I understood you, but not well enough"
that a generative model expresses naturally. Those are exactly the cases that should escalate anyway.

So the honest architecture may well be a **cascade**: encoder for the gate, encoder for field
ranking, extractive head for the value, and the generative model only for the residue the cascade
declines - with the generative model possibly not shipping at all if the residue is small enough. The
evaluation ladder in [06-eval](06-eval.md) already measures the components separately, so this is a
decision the numbers can make rather than a preference. It also changes the artefact size from
hundreds of megabytes to tens, which resolves the embedding question without ternary having to.

This is written down here rather than in a footnote because it is the most likely way this project
ends up being useful.
