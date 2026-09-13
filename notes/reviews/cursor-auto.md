# Cross-review: coddy-2bit-llm

**1. [blocker] Custom training is not yet justified against the cheaper alternatives you already named.**

The plan correctly puts R0 first, but the rest of the document talks as if the model is the product. For a closed 4-verb DSL, live schema cards, GBNF, and an exact verifier, the honest default is (a) BM25 + aliases + ~20 rules, then (b) untuned Qwen3-0.6B/1.7B with the same retrieval and grammar. Fine-tuning only earns its keep if R0 loses badly to R1 *and* untuned R1-style prompting loses badly to SFT on the *real* request mix, not on synthetic paraphrases.

Recommendation: freeze M2 until you publish R0 vs prompt-only Qwen3-0.6B vs frontier R1 on one shared eval. If prompt-only clears ship floors, cut custom training entirely and ship retrieval + grammar + optional downloaded GGUF.

**2. [major] "Mutate schemas + constrain decoder" is necessary but not sufficient for day-one unseen fields.**

What breaks is not syntax (grammar covers that). It breaks on intent→field linking when the new field's description/aliases are thin, when Russian colloquial maps to a concept never seen ("не тарахтел" → logger), when relative edits need the current value plus unit conversion, when list selectors need inventing keys (`mcp_servers[name=…]`), and when multi-field requests need planning which cards matter. Index-anchored `set #1=…` helps copy, not retrieve. A later release that adds a field with a one-line description will look like your `invented` training slice only if the retriever surfaces it; otherwise the model abstains or picks a near neighbor.

Recommendation: make retrieval recall@12 on `future-schema` a hard gate equal to generator exact-match. Add training where gold is present but buried at rank 8–12, and where two near-synonym cards compete. Prefer index-anchored outputs for the shipped model; keep literal paths as a post-decode pretty-print.

**3. [major] Track B (ternary / 1.58-bit) is not worth the critical path given your own ~1.85× number.**

At this size the story is embed size and pure-Go matmul fantasy, not speed. Distillation at ~$70–100 plus reimplementing BitNet Distillation with no released code, plus needing a Go ternary runtime that barely exists, is a second research project bolted onto a first one that has not proven product value. Q4 download-on-first-use already matches how Go CLIs ship models.

Recommendation: cut M5 from the roadmap until M0–M4 ship and users complain about 150–400 MB. If you want embeddability later, first try aggressive Q4_K_S / Q3 of a 270–360 M model before ternary.

**4. [major] GPU "$under $15" is plausible; several nearby numbers are soft or incomplete.**

SFT FLOPs order-of-magnitude for 0.6B × 180M tokens looks roughly right, and a few hours on a 4090 is believable. Weak spots: GRPO "~2e18 / $5" is hand-wavy (steps ≠ tokens; verifier-in-loop and sampling dominate wall clock); rejection-sampling cost ignores that generation is memory-bandwidth bound, not the 2ND formula; teacher data at $150–400 is the real bill and should be the headline, not GPU rent; BitNet-style "10B tokens continual pretrain" for a *narrow* task is cargo-culted from general distillation papers - you may need far less or far more, and you do not know which. Vocab/logits VRAM note is good and underplayed relative to the cheerful "fits easily" tone.

Recommendation: restate programme cost as **~$200–500 data + <$15 GPU (track A)** and treat track B as **+$100 GPU + unknown eng weeks**. Replace GRPO dollar claims with a measured 1k-step pilot before promising the ladder.

**5. [blocker] No-op / abstention as designed will not stop silent corruption in ambient mode.**

Ship criteria put no-op precision at 0.98–0.995, but ambient traffic is ~95% negatives with config-shaped vocabulary ("model", "timeout", "ключ", "порт", "faster"). Prefilter lexicon *increases* false gate triggers. "Fails twice → escalate" does not help when the model emits *valid* wrong edits (`set logger.level=error` when the user meant compile flags). Operator `config_commit` is a real control only for explicit `coddy config "…"`; ambient staging that "shows" changes will be ignored or auto-accepted by power users. "Already in that state" and "ambiguous" need calibrated confidence, not only token forcing of `NOOP`.

Recommendation: ship **explicit-only** until real-donated no-op precision is measured. Require a separate gate with precision prioritized over recall; default ambient off; never auto-stage without an explicit confirm UX. Treat valid-but-wrong as the primary hazard in eval, not parse failures.

**6. [major] Betting the Go story on goccy/go-llama (wasm→Go, ~3 months old) is fragile; the plan's fallback is too quiet.**

Risks: tokenizer mismatch (silently destroys grammar guarantees), missing or incomplete GBNF, GOAMD64/SIMD cliffs, no BitNet kernels, upstream abandonment, binary size blowups from transpiled blobs. M3's kill criterion ("cgo sub-tag, not default") is sane but undercuts the "CPU inside the Go binary / six-platform / no C toolchain" pitch that motivates track B.

Recommendation: treat go-llama as a spike only. Fallback order: (1) downloadable llama.cpp sidecar or cgo build-tag for enthusiasts; (2) external `coddy-tinyconfigd` process; (3) no on-device model - R0 rules only. Do not couple "pure Go" and "ternary" as one bet.

**7. [major] What is missing entirely.**

No threat model for prompt injection via instruction text into staged config (esp. secrets / `${ENV}` / MCP command fields). No plan for selector and list edits as first-class hard cases. No measurement of *distribution shift* between backtranslated teacher prose and real operator utterances beyond a tiny donated set. No versioning story for model↔schema↔alias triples in releases. No decision procedure for when retrieval miss should abstain vs escalate without burning the frontier path. No human baseline inter-annotator agreement on "correct edit" for ambiguous Russian. No rollback metric for "operator rejected staged suggestion" as online eval. Licensing of synthetic teacher outputs if generated on commercial APIs is waved away by "use your hub" without a hard policy.

Recommendation: add an M0.5 security+UX note (injection, secrets, confirm UX), freeze a model artefact manifest (schema hash, alias hash, model hash), and budget real-utterance collection before claiming ship floors.

**Also fine (say so plainly):**

The contract (fast path, never replacement) is sound. Targeting UCI over diffs matches the cited size regime. Verifier-as-reward is the right inductive bias. Schema-out-of-weights + runtime cards is the correct generalization strategy *directionally*. Milestone kill criteria and the R0/R1 ladder are the best part of the plan. Licence filtering of base models is done properly. Explicit-before-ambient is the right product instinct when you follow it.

## Verdict

**Do not build the custom-trained embedded SLM yet.** Build a **reduced version**: cards + hybrid retrieval + alias file + hand rules (R0), grammar-constrained **prompted** Qwen3-0.6B/1.7B behind an explicit command and optional cgo/sidecar runtime, full eval ladder including `future-schema` and adversarial no-ops. Cut track B, cut ambient-by-default, cut GRPO until SFT+RS clear a measured gap over prompt-only, and cut go-embed of weights. Revisit a fine-tuned 270–600 M artefact only if R0 and prompt-only both miss ship floors on real-donated data by a wide margin; revisit ternary only if a small Q4 model is proven useful *and* download UX is unacceptable.

