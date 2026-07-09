"""CUDA 显存分步追踪 + 逐层激活值分析。

独立 self-contained 脚本，内置 GPT 模型。不依赖项目其他代码。

用法:
    python tests/mem_profile.py --batch 16 --layers 6 --dim 384 --heads 6
    python tests/mem_profile.py configs/gpt2_lite.py --batch 16
"""

import os, sys, argparse, gc, math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 内置 GPT 模型 (self-contained)
# ═══════════════════════════════════════════════════════════════

class _Attn(nn.Module):
    def __init__(self, C, nH, T, drop=0.0):
        super().__init__()
        assert C % nH == 0
        self.nH, self.C, self.hd, self.drop = nH, C, C // nH, drop
        self.c_attn = nn.Linear(C, 3 * C, bias=True)
        self.c_proj = nn.Linear(C, C, bias=True)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.C, dim=2)
        q = q.view(B, T, self.nH, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nH, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nH, self.hd).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.drop if self.training else 0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            att = att.masked_fill(
                torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) == 0,
                float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return nn.Dropout(self.drop)(self.c_proj(y))


class _MLP(nn.Module):
    def __init__(self, C, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C, 4*C, bias=True), nn.GELU(),
                                 nn.Linear(4*C, C, bias=True),
                                 nn.Dropout(drop) if drop > 0 else nn.Identity())

    def forward(self, x): return self.net(x)


class _Block(nn.Module):
    def __init__(self, C, nH, T, drop=0.0, use_ckpt=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = _Attn(C, nH, T, drop)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = _MLP(C, drop)
        self.use_ckpt = use_ckpt

    def forward(self, x):
        if self.use_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            x = x + checkpoint(self._attn_block, self.ln1(x), use_reentrant=False)
            x = x + checkpoint(self._mlp_block, self.ln2(x), use_reentrant=False)
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x

    def _attn_block(self, x): return self.attn(x)
    def _mlp_block(self, x):  return self.mlp(x)


class _GPT(nn.Module):
    def __init__(self, V, T, L, nH, C, drop=0.0, use_ckpt=False):
        super().__init__()
        self.config = type('cfg', (), dict(V=V, T=T, L=L, nH=nH, C=C, ignore=-1))()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(V, C), wpe=nn.Embedding(T, C), drop=nn.Dropout(drop),
            h=nn.ModuleList([_Block(C, nH, T, drop, use_ckpt) for _ in range(L)]),
            ln_f=nn.LayerNorm(C)))
        self.lm_head = nn.Linear(C, V, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for blk in self.transformer.h: x = blk(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]); loss = None
        return logits, loss


# ═══════════════════════════════════════════════════════════════
# MemTracker
# ═══════════════════════════════════════════════════════════════

class MemTracker:
    def __init__(self): self.history = []

    def snap(self, phase: str):
        torch.cuda.synchronize()
        self.history.append(dict(phase=phase,
                                 alloc=torch.cuda.memory_allocated() / 1024**3,
                                 reserved=torch.cuda.memory_reserved() / 1024**3))

    def print(self, title=""):
        print(f"\n{'='*70}\n  {title}\n{'='*70}")
        print(f"  {'Phase':<52s} {'alloc':>6s} {'delta':>6s}")
        prev = self.history[0]["alloc"] if self.history else 0
        peak = max(h["alloc"] for h in self.history)
        for h in self.history:
            d = h["alloc"] - prev
            bar = _bar(h["alloc"], peak)
            mk = " <-- PEAK" if h["alloc"] == peak else ""
            print(f"  {h['phase']:<52s} {h['alloc']:5.2f}G {d:+5.2f}G {bar}{mk}")
            prev = h["alloc"]


def _bar(val, peak, w=18):
    if peak == 0: return ""
    n = int(val / peak * w)
    return "[" + "#" * n + "." * (w - n) + "]"


# ═══════════════════════════════════════════════════════════════
# 主分析
# ═══════════════════════════════════════════════════════════════

def analyze(model, B, T, steps=2, dtype=torch.float16, use_ckpt=False):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda": print("需要 CUDA"); return

    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()

    mt = MemTracker()
    mt.snap("0-empty GPU")

    if use_ckpt:
        for blk in model.transformer.h:
            blk.use_ckpt = True

    model = model.to(dev)
    mt.snap("1-model on GPU")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    mt.snap("2-optimizer + scaler")

    x = torch.randint(0, model.config.V, (B, T), device=dev)
    y = torch.randint(0, model.config.V, (B, T), device=dev)
    mt.snap("3-data on GPU")

    N = sum(p.numel() for p in model.parameters()) / 1e6

    # ── Forward hooks: 逐层看显存变化 ──
    layer_log = []

    def _hook(i):
        def _fn(m, inp, out):
            layer_log.append((f"  L{i:02d} forward", torch.cuda.memory_allocated() / 1024**3))
        return _fn

    handles = [blk.register_forward_hook(_hook(i)) for i, blk in enumerate(model.transformer.h)]

    for step in range(steps):
        layer_log.clear()
        mt.snap(f"4-step{step} before forward")

        with torch.amp.autocast("cuda", dtype=dtype):
            _, loss = model(x, y)
        mt.snap(f"5-step{step} forward done")

        # 插入逐层记录（按 alloc 排序，保持时间顺序）
        insert_idx = len(mt.history) - 1
        for name, val in layer_log:
            mt.history.insert(insert_idx, dict(phase=name, alloc=val, reserved=mt.history[-1]["reserved"]))
            insert_idx += 1

        scaler.scale(loss).backward()
        mt.snap(f"6-step{step} backward done")

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        mt.snap(f"7-step{step} optimizer step")

    for h in handles: h.remove()

    # ── 输出 ──
    mt.print(f"Memory Trace: {N:.1f}M params, B={B}, T={T}, dtype={dtype}")

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    # ── 激活值估算 ──
    L, nH, C = model.config.L, model.config.nH, model.config.C
    hd = C // nH
    items = [
        ("Q/K/V (3 tensors)", B * nH * T * hd * 3, L),
        ("Attention matrix (QK^T)", B * nH * T * T, L),
        ("Attn output", B * T * C, L),
        ("MLP hidden (4x)", B * T * 4 * C, L),
        ("MLP gelu output", B * T * 4 * C, L),
        ("MLP output", B * T * C, L),
        ("LayerNorm x2 inputs", B * T * C * 2, L),
        ("Logits", B * T * model.config.V, 1),
    ]

    print(f"\n{'='*70}")
    print(f"  Activation Estimates (B={B}, T={T}, L={L}, nH={nH}, C={C}, fp16)")
    print(f"{'='*70}")
    print(f"  {'Item':<35s} {'per layer':>8s} {'total':>8s}")
    print(f"  " + "-" * 55)
    total_est = 0
    for name, elems, count in items:
        gb = elems * 2 / 1024**3 * count  # fp16 = 2 bytes
        total_est += gb
        if gb > 0.001:
            print(f"  {name:<35s} {elems*2/1024**3:7.3f}G x{count:>2d} = {gb:7.2f}G")
    print(f"  " + "-" * 55)
    print(f"  {'Estimated total activations':<35s} {'':>8s} {total_est:7.2f}G")

    # ── 总结 ──
    param_gb = N * 2 / 1024
    grad_gb = N * 4 / 1024
    opt_gb = N * 8 / 1024
    static = param_gb + grad_gb + opt_gb
    activ = peak_alloc - static

    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  Static (params+grads+opt): {static:.2f} GiB")
    print(f"  Activations + overhead:    {activ:.2f} GiB")
    print(f"  Peak allocated:            {peak_alloc:.2f} GiB")
    print(f"  Peak reserved:             {peak_reserved:.2f} GiB")
    print(f"  Fragmentation:             {peak_reserved - peak_alloc:.2f} GiB")
    print(f"  GPU total:                 {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB")

    print(f"\n  Optimizations:")
    print(f"    Flash Attn:  no QK^T materialization  (saves {B*nH*T*T*2/1024**3*L:.1f} GiB)")
    print(f"    Grad Ckpt:   ~40% less activations     (peak ~{peak_alloc*0.6:.1f} GiB)")
    print(f"    FSDP ZeRO-3: only static {static:.1f}G -> {static/4:.1f}G  (activations unchanged)")
    print(f"    Batch=8:     activations ~halved       (peak ~{peak_alloc*0.5+static:.1f} GiB)")


def main():
    p = argparse.ArgumentParser(description="显存分步追踪 + 逐层激活分析")
    p.add_argument("source", nargs="?", help="config 文件 (可选)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--vocab", type=int, default=50304)
    p.add_argument("--dtype", choices=["float16","bfloat16","float32"], default="bfloat16")
    p.add_argument("--checkpoint", action="store_true", help="gradient checkpointing on/off")
    args = p.parse_args()

    if args.source and args.source.endswith(".py"):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import importlib.util
        spec = importlib.util.spec_from_file_location("c", os.path.abspath(args.source))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        cfg = m.model
        model = _GPT(cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.dropout)
    else:
        model = _GPT(args.vocab, args.seq_len, args.layers, args.heads, args.dim, 0.0)

    analyze(model, B=args.batch, T=model.config.T if hasattr(model.config, 'T') else args.seq_len,
            steps=args.steps, dtype=getattr(torch, args.dtype), use_ckpt=args.checkpoint)


if __name__ == "__main__":
    main()
