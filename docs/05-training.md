# 5. Training and what the hardware costs

The headline, up front, because it changes how the project should be planned:

> **GPU rent is not the cost of this project. Data is.** The supervised half - SFT, rejection
> sampling, GRPO - is single-digit to low-double-digit dollars of rented GPU time. Generating the
> training corpus is $150-400 of teacher tokens on commercial APIs, or a weekend on a self-hosted
> hub. The ternary conversion, if track B goes ahead, is another ~$100. Nothing here needs a cluster;
> everything here needs a data pipeline.

Stated as one line, so nobody plans around the wrong number:
**track A ≈ $200-500 data + under $15 GPU; track B ≈ +$100 GPU + an unknown number of engineering
weeks reimplementing a paper with no released code.**

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
| Rejection sampling, k=8 over 50 k (12 M output tokens + 60 M prefill with prefix reuse, 480 M without) | 0.6 B | see note | 1e17-6e17 |
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
| GRPO 1.5 k steps (soft) | 40 h+ | 8-24 h | 6 h | 2-4 h | ~$5 |
| **Whole SFT+RS+GRPO programme** | - | **~1.5 days** | - | - | **under $15** |

Rental rates sampled 2026-08-11: RunPod community 4090 $0.34/h, A100-80 $1.19/h, H100 SXM $2.69/h;
Vast.ai 4090 $0.34-0.50, verified H100 $1.50-1.87.

Two of these rows are softer than the others and are marked as such. **GRPO** is priced from a step
count, and steps are not tokens: the wall clock is dominated by sampling k completions per prompt and
by running the verifier in the loop, neither of which the 6ND formula sees. **Rejection sampling** is
priced with the 2ND inference formula, which assumes compute-bound batched generation; at small batch
sizes it is memory-bandwidth bound instead and the real number can be several times worse. Both get a
measured 100-step pilot before anyone plans around them.

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
Qwen3-0.6B saw thousands of tokens per parameter. Taking it to 100 B tokens is 1.56e20 FLOPs - about **16 h on
8×H100** (~$205) or eight days on one 4090 - and it still lands below a fine-tuned 0.6 B.

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

One thing to question rather than copy: the 10 B token warm-up is BitDistill's figure for restoring
*general* capability in a general model. This model is narrow, bilingual and grammar-constrained, and
it is entirely possible that a fraction of that suffices - or that a narrow model needs relatively
more, because it has less redundancy to lose. Nobody has published the answer. Treat 10 B as the
budget ceiling and run the ablation (1 B / 3 B / 10 B tokens) rather than paying it by default.

So: **the ternary version costs about a hundred dollars and a week of one desktop GPU**, on top of a
track A model that already works. What it returns is a file at 2.4-3.7 bits per weight rather than
4.85 - see [03-architecture](03-architecture.md) §3.3 for why that is not the 1.58 the name implies. That is the number that decides track B, and it is small. What it
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
