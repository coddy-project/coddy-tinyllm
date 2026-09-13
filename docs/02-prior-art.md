# 2. Prior art

What is already known, what is measured, and what nobody has published. Every number here comes
from a named source; where a number is derived rather than reported, it says so.

## A. The article this started from

[Тернарная нейронка на C#](https://habr.com/ru/articles/1074678/) walks through why a ternary
weight is the smallest *complete* neuron (it can amplify, attenuate and stay silent, where a binary
weight can only do two of the three), explains the BitNet b1.58 training trick - BF16 master weights,
per-layer threshold equal to the mean absolute weight, straight-through estimator, ternary weights
re-materialised on every forward pass - and then builds a ternary Llama in C# to show it end to end.

The code is [virex-84/LLamaTritLLM](https://github.com/virex-84/LLamaTritLLM): MIT, .NET 8, ~6.3k
lines, one commit. Two projects, `TernaryTraining` (hand-written autograd, AdamW, BPE, QAT with STE
over FP32 latent weights) and `TernaryInference`. The packing format is the interesting part -
multi-plane ternary, `w ≈ Σ_p scale[p]·t[p]`, five trits per byte in base 3, FP16 per-group scales,
so `planes=1` is 1.58 bit and more planes trade size for capacity.

The scale is the part to be clear-eyed about: the shipped configurations are **106 496 and 49 152
parameters** (embedding 64/32, one layer, one head), the models are 28.9 KB and 17.1 KB, and the
training corpus is 22 Russian sentences in a 4 KB JSON file. The inference kernel is a scalar
per-row dot product - no SIMD, no int8 path, no parallelism.

**Verdict: a readable reference for trit packing and STE, and nothing else.** It is not a base to
build on, not a performance reference, and not a source of a usable model. Its real contribution to
this project is the reminder that the BitNet recipe is small enough that one person can implement it.

## B. Ternary and 1.58-bit in September 2026

### What exists and runs

[microsoft/BitNet](https://github.com/microsoft/BitNet) (bitnet.cpp, MIT, ~40 k stars) is
**inference only** - there is no training code in the tree. Three kernels: **I2_S** (2 bpw, plain
multiply-add GEMV), **TL1** (LUT, two weights into a 4-bit index), **TL2** (three weights into
5 bits, 1.67 bpw). Support matrix for b1.58-2B-4T: x86 gets I2_S and TL2, ARM gets I2_S and TL1.

Downloadable ternary weights, the complete list:

| Model | Size | Tokens | Licence |
|---|---|---|---|
| [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | 2.4 B | 4 T | MIT |
| [BitNet-embedding-0.6B / 270M](https://huggingface.co/microsoft/BitNet-embedding-0.6B) | 0.6 B / 270 M | - | MIT, **embedding only** |
| [SpectraSuite TriLM](https://huggingface.co/SpectraSuite) 99 M … 3.9 B | 99 M-3.9 B | 300 B | Apache 2.0, shipped unpacked FP16 |
| [Falcon-E 1B/3B](https://falcon-lm.github.io/blog/falcon-edge/) | 1 B / 3 B | ~1.5 T | Falcon-LLM licence |
| [Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) | 8 B | 100 B converted | Llama 3 |

There is **no sub-500M generative ternary chat model in existence.** The only small ternary
generative weights are TriLM 99M/190M/390M, which are 2024-vintage 300B-token general LMs.

### How fast it actually is at our size

The number that matters, from bitnet.cpp's own paper ([arXiv:2502.11880](https://arxiv.org/abs/2502.11880),
Table 7), unlimited threads:

| CPU | model | FP16 | Q4_0 | I2_S | TL2 |
|---|---|---|---|---|---|
| i7-13700H | 700 M | 30.7 | 67.6 | **125.4** | 127.0 |
| Apple M2 | 700 M | 110.7 | 197.4 | **238.2** | 229.2 |

So at 700 M, **ternary is ~1.85× faster than Q4_0, not 6×.** The 6.17× headline compares against
FP16 at larger sizes. The memory ratio is the stronger claim: b1.58-2B-4T runs in 0.4 GB
non-embedding memory against 1.4-2.6 GB for Gemma-3-1B and Qwen2.5-1.5B, at 29 ms/token CPU decode
and 0.028 J/token against 0.186-0.347 J.

### The scaling law, and why from-scratch is the wrong plan

[Spectra](https://arxiv.org/abs/2407.12327) fits `L(N) = A/N^0.26 + ε` at a fixed 300 B tokens, with
A = 185, ε = 1.76 for ternary against A = 159, ε = 1.67 for float. Solving that for equal loss gives
the parameter multiplier a ternary model needs: **2.23× at 100 M, 2.39× at 300 M, 2.64× at 1 B,
3.5× at 10 B**. Ternary only approaches float within 6-7 % somewhere above 15 B parameters. The same
paper notes that models below 1 B saw training loss *increase* at the mid-training learning-rate
drop - small ternary runs are the unstable ones. [Spectra-1.1](https://arxiv.org/abs/2506.23025)
adds that TriLMs are data-hungry: extra tokens buy more than extra parameters.

The one published small-and-narrow from-scratch data point,
[TernaryLM](https://arxiv.org/abs/2602.07374) (132 M on TinyStories), gets validation perplexity
**58.4 against 28.3** for the identical FP32 architecture on the same data - 2.07× worse, consistent
with the scaling law.

### The path that does work: distillation

[BitNet Distillation](https://arxiv.org/abs/2510.13998) (Microsoft, Oct 2025) converts an existing
FP16 model into a 1.58-bit one in three stages: SubLN inserted before the attention and FFN output
projections, ~10 B tokens of continual pretraining, then fine-tuning with logits KL (τ=5) plus
MiniLM-style attention-relation distillation on a single layer, teacher = the same model SFT'd.

The results are the reason track B is in this plan at all, because they are measured **on narrow
tasks at our exact size**:

| Model, MNLI | FP16 SFT | naive 1.58-bit SFT | BitDistill |
|---|---|---|---|
| Qwen2.5-0.5B | 79.91 | 60.80 | **79.98** |
| Qwen3-0.6B (avg MNLI/QNLI/SST-2) | 88.01 | 74.09 | **88.17** |

CPU throughput 427 -> 1135 tokens/s at 16 threads, memory 1.20 -> 0.11 GB. Without the continual
pretraining stage the naive gap *widens* with model size.

Two caveats worth carrying forward. The paper points at microsoft/BitNet for code, and that repo has
no training code - **no BitDistill implementation appears to have been released**, so this is a
reimplementation, not a `pip install`. And TII states the negative result plainly in
[onebitllms](https://github.com/tiiuae/onebitllms): fine-tuning an arbitrary existing checkpoint into
BitNet format "often leads to poor performance"; only pre-quantised checkpoints are supported, full
fine-tune only, LoRA unsupported, ~20 % training overhead. The continual-pretraining stage is not
optional.

LoRA *on top of* an already-ternary base does work - [QVAC/Tether](https://huggingface.co/blog/qvac/fabric-llm-finetune-bitnet)
fine-tuned a 125 M BitNet in ~10 minutes on a Galaxy S25.

### Cost, honestly labelled

**No vendor has published GPU-hours for any ternary model.** Microsoft's
[2B-4T report](https://arxiv.org/abs/2504.12285) gives the data mix and the LR schedule and no
hardware. Falcon-E gives no compute figure. Spectra says V100 nodes and no hours. Everything in
[05-training](05-training.md) §5.5 is therefore derived from 6ND, not reported.

## C. Small models on narrow structured tasks

This is where the evidence is strongest, and it is what justifies the whole project.

- **Kubernetes manifests from natural language**, the closest published analogue:
  [Context-Instrumental Data Distillation](https://arxiv.org/html/2605.25835v1) fine-tunes
  Qwen2.5-Coder-1.5B and reaches **full-pass@1 91.5 %** where full pass means syntax +
  `kubeconform --strict` + semantic + security checks. **1 000 training examples already gave 91.0 %.**
- **Tool calling at 350 M**: [distil labs](https://www.distillabs.ai/blog/fine-tuning-liquids-lfm25-accurate-tool-calling-at-350m-parameters/)
  took shell-command tool-call accuracy from **61.4 % to 98.0 %** on 5 000 synthetic examples,
  beating its own GPT-oss-120B teacher (97.0 %).
- **Ansible YAML**: WISDOM-ANSIBLE-MULTI at **350 M beat CodeGen-6B by ~15 points** on their Ansible
  Aware metric ([paper](https://arxiv.org/html/2402.17442v1)) - domain-specialised sub-1B beating a
  generic 6B on YAML generation.
- **Structured output with RL**: [ThinkJSON](https://arxiv.org/html/2502.14905v1) trains Qwen2.5-1.5B
  with GRPO to 62.41 % mean match / 0.27 % noise, beating DeepSeek-R1-671B (41.43 % / 11.14 %).

Counterweight, for calibration: [IaC-Eval](https://openreview.net/pdf?id=7TCK0aBL1C) has GPT-4 under
**20 % pass@1** on 458 Terraform tasks, and [NetConfEval](https://dl.acm.org/doi/10.1145/3656296) /
[SLM_netconfig](https://arxiv.org/abs/2512.02861) report 57-74 % on network configs. Configuration
*generation* from scratch is hard. Configuration *editing* against a retrieved schema slice is a much
narrower problem, which is the bet this project makes.

## D. How to represent an edit

The decisive evidence is [Diff-XYZ](https://arxiv.org/html/2510.12487v2), which evaluates udiff,
udiff-h, udiff-l and search/replace across Qwen2.5-Coder from 0.5 B to 32 B:

| Model | best diff-format apply EM |
|---|---|
| 0.5 B | **0.00-0.01** on every format |
| 1.5 B | 0.22 (search/replace), 0.16 (udiff) |
| 3 B | 0.41 / 0.38 |

Their conclusion: "smaller open models benefit little from any formatting choice". **A line diff is
not an option at our size.** [Aider](https://aider.chat/docs/more/edit-formats.html) measured the
same axis for large models and found whole-file and search/replace comparable, with search/replace
cheaper in tokens; their `diff-fenced` format exists only because one model mishandles fencing, which
is itself a warning that edit formats are model-specific.

[JSON Whisperer](https://arxiv.org/html/2510.04717v1) (EMNLP 2025) generates RFC 6902 patches at
**31 % fewer tokens** than full regeneration with quality within 5 %, and names the failure mode:
**array index arithmetic** - index shifts, zero-versus-one base. Their fix (EASE) replaces indices
with stable keys.

Coddy's existing UCI command language is EASE's conclusion taken further: one line per edit, stable
dotted addressing, `name[key=value]` instead of indices, no context matching, and a parser and
dry-run already written. Nothing in the literature suggests a better target for a sub-1B model.

**Gap: nobody has published a head-to-head of DSL against JSON Patch against whole-file rewrite at
sub-1B scale.** We will have to measure it ourselves; [06-eval](06-eval.md) plans for it.

## E. Generalising to a schema the model was not trained on

- [Schema-Guided Dialogue](https://arxiv.org/pdf/1909.05855) established the pattern: describe slots
  in natural language so a model can generalise to unseen APIs.
- [SGD-X](https://ar5iv.labs.arxiv.org/html/2110.06800) is the warning label: models turn out to be
  highly sensitive to schema surface form, and **71 % of "unseen" intent names and 65 % of "unseen"
  slot names actually appeared in training**, so the published zero-shot numbers were inflated. This
  is precisely the trap [04-data](04-data.md) §4.3 builds schema mutation to avoid.
- **Text-to-SQL schema linking is the mature analogue**: retrieve a slice, keep recall high, tolerate
  over-inclusion ([RSL-SQL](https://arxiv.org/pdf/2411.00073),
  [extractive schema linking](https://arxiv.org/pdf/2501.17174),
  [LinkAlign](https://aclanthology.org/2025.emnlp-main.51.pdf)). The dissenting
  ["Death of Schema Linking?"](https://arxiv.org/abs/2408.07702) argues strong reasoners no longer
  need it - which does not apply at 0.6 B, where retrieval is the mechanism rather than an
  optimisation.
- Quantitative support from config-land: knowledge injection lifted IaC generation from **27.1 % to
  62.6 %**, with Graph RAG reaching 80.3 % technical validation against 37.2 % baseline
  ([paper](https://arxiv.org/abs/2512.14792)) - and the same paper's caveat, that structured
  knowledge fixes validity but not intent alignment.

**Gap: no published work on key-path retrieval over a YAML config schema, or field-description
embedding for config keys.**

## F. Saying "this is not about settings"

- [AbstentionBench](https://arxiv.org/html/2506.09038v1) across 20 datasets: **scale does not help**
  (Llama 3.1 shows "almost no effect" from 8 B to 405 B), and reasoning fine-tuning *costs* about
  24 % abstention on average. A tuned system prompt does help without hurting precision.
- **A separate classifier beats asking the generator.** [UDRIL](https://aclanthology.org/2025.acl-industry.25.pdf)
  puts a DistilBERT in front and routes only the 5-15 % most uncertain cases to the LLM, reaching
  **F1 0.761 against 0.566 for Mistral-7B alone** (+34 % relative). [DROID](https://arxiv.org/pdf/2510.14110)
  reports state-of-the-art out-of-scope F1 with a frozen MiniLM encoder.

This is the direct justification for the separate gate in [03-architecture](03-architecture.md) §3.1,
and for never asking the generator to decide whether it should run.

## G. Constrained decoding

All the serious implementations compile a schema into a decoding constraint **at runtime**, which is
what makes "the schema changes without retraining" mechanically true rather than aspirational.

- [XGrammar-2](https://arxiv.org/html/2601.04426): most static JSON Schemas compile in **under 1 ms**,
  dynamic tool-call schemas in ~10 ms, per-token mask overhead ~13 µs mean, 44-48 µs p99.
- [llguidance](https://github.com/guidance-ai/llguidance): Rust, ~50 µs/token at a 128 k vocabulary,
  with a C header - the practical route into Go via cgo. No first-party Go binding.
- [JSONSchemaBench](https://arxiv.org/html/2501.10868v3) over 10 k real schemas measures compile
  times: Guidance 0.00-0.01 s, llama.cpp 0.05-0.06 s, XGrammar 0.12-0.30 s, **Outlines 3.48-8.05 s**
  (avoid for per-request schemas). Constrained decoding *improved* downstream accuracy by up to 4 %
  (GSM8K 80.1 -> 83.8).
- The caveats that keep this honest: ["Let Me Speak Freely"](https://arxiv.org/pdf/2408.02442) reports
  format restriction degrading reasoning on some task mixes, and
  ["Where vs What"](https://arxiv.org/html/2608.25358) shows that a grammar fixes format only - at
  depth-4 nesting Qwen2.5-7B still **misplaces 73.8 %** of recalled values. **A grammar buys
  validity, never correctness.**

For Go specifically, the question mostly dissolves: if the runtime is goccy/go-llama, GBNF goes
straight to llama.cpp's own grammar engine and no Go grammar implementation is needed. If we ever
write our own runtime, the only serious pure-Go option is
[ThiraSoft/golem `grammar/`](https://github.com/ThiraSoft/golem/tree/main/grammar), which ports
`llama-grammar.cpp` into an automaton that handles UTF-8 split across byte-BPE tokens and refuses
invisible tokens and mid-rule EOG, plus a `json-schema-to-grammar.cpp` port tested line-for-line
against llama.cpp's expected output - on a one-star, single-author repository.
[gbnf-go](https://github.com/sbellity/gbnf-go) is zero stars, created and last pushed the same day,
and is not a dependency. Note also that a pure-Go **GGUF writer** barely exists (one 6.7 KB file in
golem); `convert_hf_to_gguf.py` stays the packaging step, which is fine - packaging happens on a
developer machine, not on the user's.

## H. Running a small model from Go

Verified against the GitHub API on 2026-09-13.

| Option | Stars / activity | cgo | Verdict |
|---|---|---|---|
| [goccy/go-llama](https://github.com/goccy/go-llama) | 162, pushed 2026-09-09, MIT | **no** | llama.cpp compiled to wasm and then *transpiled to Go source* via [wasm2go](https://github.com/goccy/wasm2go), so nothing runs a wasm VM at runtime. GGUF, embeddings, **grammars**, LoRA. Reported 205 tok/s for 0.5 B Q8_0 at 8 threads on Apple silicon. Three months old. |
| [hybridgroup/yzma](https://github.com/hybridgroup/yzma) | 616, pushed 2026-09-13 | no (purego + ffi) | real llama.cpp at full speed, but ships or downloads a shared library per platform |
| [onnxruntime_go](https://github.com/yalue/onnxruntime_go) / [hugot](https://github.com/knights-analytics/hugot) | 714 / 646 | yes | mature, needs the ORT shared library per target |
| [gomlx](https://github.com/gomlx/gomlx) simplego | 1 634, Apache 2.0 | no | pure-Go backend ~5× slower than XLA |
| [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp) | 940, dead since 2024-03 | yes | do not use |
| hand-written Go kernels | - | no | Go 1.26 ships `simd/archsimd` behind `GOEXPERIMENT=simd`, **amd64 only**; the enable-by-default proposal ([#78979](https://github.com/golang/go/issues/78979)) is on hold and arm64/SVE has no timeline. **No ternary kernel exists in Go at all.** |

Ollama is no longer a source of vendorable Go inference:
[PR #16031](https://github.com/ollama/ollama/pull/16031) (merged 2026-05-29) removed the vendored
GGML and the Go model implementations in favour of upstream `llama-server`. What survives and is
worth taking is [`ollama/ollama/fs/ggml`](https://pkg.go.dev/github.com/ollama/ollama/fs/ggml), a
pure-Go GGUF metadata reader (MIT).

Supporting pieces: [trengrj/go-potion](https://github.com/trengrj/go-potion) (MIT, pure Go,
model2vec/potion static embeddings, vectors identical to the Python original, ~60 MB/s on an M1 Max)
is exactly the retrieval half of [03-architecture](03-architecture.md) §3.2;
[sugarme/tokenizer](https://github.com/sugarme/tokenizer) (Apache 2.0, 334 stars) is the only
established pure-Go `tokenizer.json` loader, and token-ID parity against Hugging Face must be
verified rather than assumed.

### Ternary through a normal GGUF runtime

The finding that decides whether track B is even buildable: **ternary is in mainline llama.cpp, not
only in the Microsoft fork.** [PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151) merged
2024-09-06 adding `TQ1_0` (1.6875 bpw) and `TQ2_0` (2.0625 bpw) to ggml, with x86 AVX2 and ARM NEON
kernels complete at merge. The PR reports bandwidth rather than tokens/s - TQ2_0 at 141.83 GB/s
f32-equivalent on a Core m3-8100Y, roughly 2x Q4_K. **No published tokens/s for TQ2_0 was found.**

Two caveats. `microsoft/bitnet-b1.58-2B-4T-gguf` ships **I2_S**, which upstream cannot load
([issue #12997](https://github.com/ggml-org/llama.cpp/issues/12997) closed with "use ik_llama.cpp or
bitnet.cpp") - the working path is converting the safetensors master with
`convert_hf_to_gguf.py --outtype tq2_0`, which we would be doing anyway for our own model. And the TQ
types have `vec_dot` only, no repacked GEMM, so upstream does not reproduce bitnet.cpp's TL1/TL2
speedups - prefill in particular stays slower than the numbers in the table above.

And the coverage check that matters, read out of goccy/go-llama v0.5.0's own embedded kernel index
(`base/asm_kernels.json`, 49 kernels, identical on amd64 and arm64):

- repacked `gemm` + `gemv` + `vec_dot`: `q4_0, q4_K, q5_0, q5_K, q6_K, q8_0, iq4_nl, mxfp4`;
- `vec_dot` only: `q2_K, q3_K, q4_1, q5_1, iq1_s/m, iq2_*, iq3_*, iq4_xs, nvfp4, q1_0, q2_0,
  **tq1_0, tq2_0**`.

So both tracks target the same Go runtime, and track B stops being an inference research project.

### What adopting goccy/go-llama actually costs

Measured locally on 2026-09-13 with Go 1.26.8, `CGO_ENABLED=0`, no weights (empty Go binary
baseline 1.6 MB):

| target | binary | build |
|---|---|---|
| linux/amd64 `GOAMD64=v2` | 20.2 MB | 28 s |
| linux/amd64 `GOAMD64=v1` | 21.7 MB | 2 m 28 s |
| darwin/arm64 (cross) | 17.9 MB | 23 s |
| windows/amd64 v2 | 20.4 MB | fast |

All cross-compiled from linux/amd64 with no toolchain setup - the claim that this keeps the release
pipeline unchanged holds. Grammar support is real: `Params.Grammar` takes plain GBNF straight to
llama.cpp's own grammar engine.

**`GOAMD64=v2` is mandatory, not advisory.** The project's own toolchain check states that at v1
"the engine bundle has no asm at all ... the pure-Go scalar fallback (~1000x slower; the big-model
suite times out)". That is a packaging decision for Coddy: the tagged build has to set
`GOAMD64=v2`, which raises the amd64 floor to SSE4.2-era hardware (2009 and later). The transpiled
engine itself is a separate MIT module,
[llamawasm2go](https://github.com/goccy/llamawasm2go), 133 MB in the module cache - an ordinary
`go.mod` dependency, nothing generated at build time, but it is 133 MB in any vendor directory.

**No Go CLI in 2026 ships an embedded generative LLM.** The largest ship-the-weights precedent is a
~90 MB embedding model, and it is cgo. We would be early.

## I. Russian under 1 B

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): 119 languages, 36 T pretraining tokens,
  Apache 2.0 - the default choice.
- **SmolLM3 is disqualified**: [six languages officially](https://huggingface.co/blog/smollm3), no
  Russian.
- Gemma 3 270M claims 140+ languages family-wide with a 256 k vocabulary, but no per-size language
  breakdown is published. Treat it as a fine-tuning substrate, not a Russian instruction follower.
- Russian-specialised: [Vikhr-Qwen-2.5-0.5B-Instruct](https://huggingface.co/Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct)
  (SFT on 150 k Russian instructions) and [RuadaptQwen2.5-1.5B](https://huggingface.co/RefalMachine/RuadaptQwen2.5-1.5B-instruct),
  which retrains the tokenizer for Russian - relevant because Russian token count is latency here.
- Benchmarks: [MERA](https://mera.a-ai.ru/), [ru_llm_arena](https://github.com/VikhrModels/ru_llm_arena).
  The [Gamayun](https://arxiv.org/pdf/2512.21580) paper reports MERA ≈ 36.4 for Qwen3-1.7B, 35.7 for
  Qwen2.5-1.5B - near-floor territory, so aggregate Russian benchmarks at this size say little.
  **Gap: no ruIFEval-style instruction-following score exists for any sub-1B model.** We measure our
  own.

## J. What nobody has published

These are simultaneously the project's contribution and its risk register:

1. DSL versus JSON Patch versus whole-file rewrite, measured at sub-1B scale;
2. key-path retrieval over a configuration schema, with the text-to-SQL recall metric;
3. Russian instruction following under 1 B, measured on a task rather than a leaderboard;
4. a ternary model distilled to a narrow *generative* task at 300-600 M - BitDistill published
   classification and summarisation, not constrained command generation.
