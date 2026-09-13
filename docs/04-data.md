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
seed for the intent taxonomy**, not a training corpus. And the negative class is abundant but
**noisily labelled**: a turn that did not lead to a config edit is usually a non-config turn, but it
may equally be one where the agent missed the intent, took another route, failed, or was interrupted.
This is positive-unlabeled data, and treating it as clean negatives would teach the model to abstain
on exactly the requests it should serve. A stratified manual audit of a sample, and a PU-aware or
confidence-weighted objective, are the price of using it.

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
the model learns that the answer is always present and will confabulate when retrieval misses. Three
deliberate slices, none of which a naive generator would produce:

- **gold absent** (target 15-20 %): the correct answer is abstention;
- **gold buried** (target 20 %): the right field sits at rank 8-12, not rank 1, so the model cannot
  learn "take the first card";
- **near-synonym competition** (target 15 %): two or three cards plausibly match the request
  (`agent.max_turns` against `agent.max_tokens_per_turn`, `logger.level` against
  `logger.levels[]`), and only the current values or the phrasing disambiguate.

The list and selector cases (`add_list`, `del_list`, `mcp_servers[name=x].command`) are their own
difficulty class and get their own quota rather than being left to chance: they are where the address
has to be *constructed* from the config's current contents rather than copied from a card.

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

Russian and English at a 50/50 split, held to that ratio deliberately rather than allowed to drift,
with per-language floors in the ship criteria: an aggregate number hides a Russian false-edit rate
twice the English one, and Russian is the half more likely to go unnoticed by a reviewer reading
English. The Russian half also carries the harder variation
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
