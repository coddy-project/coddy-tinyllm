"""Measure real CPU training throughput on a Llama-shaped block stack.

Answers one question: can this be trained without a GPU? Run it on the machine you
would use, then divide 6*N*D by the effective GFLOPS it reports.

    uv venv .venv && uv pip install --python .venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch
    .venv/bin/python train_bench.py

Measured on an i7-8750H (6 cores / 12 threads, AVX2, no AVX-512, torch 2.14+cpu),
2026-09-13:

    qwen3-0.6B-like, 4 layers  params  210.1M   9473 ms/step  108 tok/s  136 GFLOPS
    smollm2-135M-like, 4 lay.  params   44.2M   2145 ms/step  477 tok/s  127 GFLOPS
    encoder-ish 30M, 6 layers  params   23.2M    887 ms/step  577 tok/s   80 GFLOPS

For reference, a plain 2048^3 sgemm on the same machine reaches 217 GFLOPS with
OpenBLAS, so training runs at roughly 60 % of the matmul ceiling. See
../../docs/05-training.md for what those numbers mean in days.
"""
import time, math, torch, torch.nn as nn

torch.set_num_threads(12)

class Block(nn.Module):
    def __init__(s, d, ffn, heads):
        super().__init__()
        s.h = heads; s.dh = d // heads
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)
        s.g = nn.Linear(d, ffn, bias=False); s.u = nn.Linear(d, ffn, bias=False)
        s.dn = nn.Linear(ffn, d, bias=False)
        s.n1 = nn.RMSNorm(d); s.n2 = nn.RMSNorm(d)
    def forward(s, x):
        B, T, D = x.shape
        y = s.n1(x)
        q, k, v = s.qkv(y).chunk(3, -1)
        q, k, v = (t.view(B, T, s.h, s.dh).transpose(1, 2) for t in (q, k, v))
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        y = s.n2(x)
        return x + s.dn(torch.nn.functional.silu(s.g(y)) * s.u(y))

class Model(nn.Module):
    def __init__(s, vocab, d, ffn, heads, layers):
        super().__init__()
        s.emb = nn.Embedding(vocab, d)
        s.blocks = nn.ModuleList(Block(d, ffn, heads) for _ in range(layers))
        s.nf = nn.RMSNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
        s.head.weight = s.emb.weight
    def forward(s, idx):
        x = s.emb(idx)
        for b in s.blocks: x = b(x)
        return s.head(s.nf(x))

def bench(label, vocab, d, ffn, heads, layers, T, steps=3, B=1):
    m = Model(vocab, d, ffn, heads, layers)
    n = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=False)
    idx = torch.randint(0, vocab, (B, T))
    tgt = torch.randint(0, vocab, (B, T))
    # warm-up
    loss = nn.functional.cross_entropy(m(idx).view(-1, vocab), tgt.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    t = time.time()
    for _ in range(steps):
        loss = nn.functional.cross_entropy(m(idx).view(-1, vocab), tgt.view(-1))
        loss.backward(); opt.step(); opt.zero_grad()
    dt = (time.time() - t) / steps
    toks = B * T
    flops = 6 * n * toks
    print(f"{label:<26} params {n/1e6:6.1f}M  {dt*1000:8.0f} ms/step  "
          f"{toks/dt:7.0f} tok/s  {flops/dt/1e9:6.1f} GFLOPS effective")
    return toks / dt, n

# Qwen3-0.6B shape, but 4 layers instead of 28 (extrapolate linearly on the block stack)
bench("qwen3-0.6B-like, 4 layers", 151936, 1024, 3072, 16, 4, T=1024)
bench("smollm2-135M-like, 4 lay.", 49152, 576, 1536, 9, 4, T=1024)
bench("encoder-ish 30M, 6 layers", 32768, 384, 1024, 6, 6, T=512)
