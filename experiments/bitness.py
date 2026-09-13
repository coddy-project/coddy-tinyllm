#!/usr/bin/env python3
"""What bit-width does a "1.58-bit" model actually land at?

Ternary quantisation applies to the linear layers only. The embedding and the
output head stay 4-8 bit, and in a small model with a large multilingual
vocabulary they are a large share of the parameters - which is what decides the
real bits-per-weight of the file on disk.

Configs read from each model's config.json on 2026-09-13; Gemma 3 270M is gated,
so its split is Google's published figure.
"""

MODELS = {
    # name: (vocab, hidden, layers, intermediate, heads, kv_heads, head_dim)
    "Qwen3-0.6B":     (151936, 1024, 28, 3072, 16, 8, 128),
    "Qwen3-1.7B":     (151936, 2048, 28, 6144, 16, 8, 128),
    "SmolLM2-360M":   (49152, 960, 32, 2560, 15, 5, 64),
    "SmolLM2-135M":   (49152, 576, 30, 1536, 9, 3, 64),
}
# bits per weight, as stored by llama.cpp (scales included)
BPW = {"TQ1_0": 1.6875, "TQ2_0": 2.0625, "Q4_0": 4.5, "Q4_K_M": 4.85, "Q8_0": 8.5, "F16": 16.0}


def split(vocab, hidden, layers, inter, heads, kv, hd):
    """(embedding params, linear-layer params). Tied embeddings counted once."""
    emb = vocab * hidden
    q = hidden * heads * hd
    k = hidden * kv * hd
    v = k
    o = heads * hd * hidden
    mlp = 3 * hidden * inter
    return emb, layers * (q + k + v + o + mlp)


def mb(bits):
    return bits / 8 / 1024 / 1024


def row(name, cfg, body_q, emb_q, vocab_override=None):
    vocab, hidden, layers, inter, heads, kv, hd = cfg
    if vocab_override:
        vocab = vocab_override
    emb, body = split(vocab, hidden, layers, inter, heads, kv, hd)
    total = emb + body
    bits = emb * BPW[emb_q] + body * BPW[body_q]
    return {
        "name": name,
        "params_M": total / 1e6,
        "emb_share": emb / total,
        "file_MB": mb(bits),
        "avg_bpw": bits / total,
    }


def main():
    print("Parameter split (tied embeddings counted once)\n")
    print(f"{'model':<16}{'total M':>9}{'embed M':>9}{'linear M':>10}{'embed share':>13}")
    for name, cfg in MODELS.items():
        emb, body = split(*cfg)
        print(f"{name:<16}{(emb+body)/1e6:>9.0f}{emb/1e6:>9.0f}{body/1e6:>10.0f}{emb/(emb+body):>12.0%}")

    print("\nFile size and true average bits per weight\n")
    schemes = [
        ("all Q4_K_M (track A)", "Q4_K_M", "Q4_K_M", None),
        ("TQ2_0 body + Q8 embed", "TQ2_0", "Q8_0", None),
        ("TQ2_0 body + Q4 embed", "TQ2_0", "Q4_0", None),
        ("TQ1_0 body + Q4 embed", "TQ1_0", "Q4_0", None),
    ]
    for name, cfg in MODELS.items():
        print(f"  {name}")
        for label, bq, eq, vo in schemes:
            r = row(name, cfg, bq, eq, vo)
            print(f"    {label:<24}{r['file_MB']:>8.0f} MB   {r['avg_bpw']:>5.2f} bpw average")
        if cfg[0] > 60000:
            r = row(name, cfg, "TQ2_0", "Q8_0", 32768)
            print(f"    {'+ vocab trimmed to 32k':<24}{r['file_MB']:>8.0f} MB   {r['avg_bpw']:>5.2f} bpw average")
        print()

    print("Gemma 3 270M, from Google's published split (170 M embedding, 100 M transformer,")
    print("256 k vocabulary) - gated repo, not read from config.json:")
    emb, body = 170e6, 100e6
    for label, bq, eq in (("all Q4_K_M", "Q4_K_M", "Q4_K_M"), ("TQ2_0 + Q4 embed", "TQ2_0", "Q4_0")):
        bits = emb * BPW[eq] + body * BPW[bq]
        print(f"    {label:<24}{mb(bits):>8.0f} MB   {bits/(emb+body):>5.2f} bpw average")


if __name__ == "__main__":
    main()
