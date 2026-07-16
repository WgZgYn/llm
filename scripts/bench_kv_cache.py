"""KV-Cache 加速效果基准测试 —— 逐 token 延迟 vs 序列长度。

用法:
    python scripts/bench_kv_cache.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm.model.gpt import GPT
from llm.config.model_config import GPTConfig


def bench(cfg, name, prompt_len=32, n_tokens=200, device='cuda'):
    """测试有/无 KV-Cache 的逐 token 延迟。"""
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"  [WARN] CUDA 不可用，回退到 CPU。")
        device = 'cpu'
    model = GPT(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size or 1000, (1, prompt_len), device=device)
    n_params = model.get_num_params(non_embedding=False) / 1e6
    sync = torch.cuda.synchronize if device == 'cuda' else (lambda: None)

    # 预热
    for _ in range(5):
        model.generate(prompt, 10, use_cache=False)
        model.generate(prompt, 10, use_cache=True)
    sync()

    # ── 无 cache: 测每个 token 的延迟 ──
    # 注意: 这里直接调用 model() 前向 + argmax（贪婪解码），
    # 排除了 softmax/multinomial 采样开销，仅测量前向延迟。
    idx = prompt.clone()
    times_nocache = []
    for _ in range(n_tokens):
        idx_cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]
        t0 = time.perf_counter()
        logits, _ = model(idx_cond)
        sync()
        t1 = time.perf_counter()
        idx = torch.cat([idx, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        times_nocache.append((t1 - t0) * 1000)
        # 收集 40 个点即可（趋势已明显）
        if len(times_nocache) >= 40:
            break
    n_nocache = len(times_nocache)

    # ── KV-Cache ──
    # 注意: avg_cache 包含首 token 的 prefill (完整 prompt 的前向)，
    # 后续 token 才享受 cache 加速。因此 avg_cache 会略微高估逐 token 延迟。
    # speedup 使用 "最后10个无 cache token" vs "cache 平均" 以凸显长序列差距。
    t0 = time.perf_counter()
    model.generate(prompt, n_nocache, use_cache=True)
    sync()
    t_total_cache = time.perf_counter() - t0
    avg_cache = (t_total_cache / n_nocache) * 1000

    # ── 打印 ──
    avg_all = sum(times_nocache) / n_nocache
    avg_no_last10 = sum(times_nocache[-10:]) / 10
    print(f"\n{name}  ({n_params:.1f}M params, {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}D)")
    print(f"  序列从 {prompt_len} → {prompt_len + n_nocache} tokens")
    print(f"  无 cache:  avg={avg_all:5.1f}ms/token  "
          f"最后10个={avg_no_last10:5.1f}ms/token")
    print(f"  KV-Cache:  avg={avg_cache:5.1f}ms/token (含 prefill)")
    speedup = avg_no_last10 / avg_cache if avg_cache > 0 else 0
    print(f"  加速比 (最后10 token vs cache avg): {speedup:.1f}x")
    print(f"  首 token 延迟 (无cache): {times_nocache[0]:.1f}ms")


if __name__ == "__main__":
    print("=" * 55)
    print("KV-Cache Bench: 逐 token 推理延迟")
    print("=" * 55)

    # 小模型短序列
    bench(GPTConfig(vocab_size=1000, block_size=512,
                    n_layer=8, n_head=8, n_embd=256, dropout=0.0, bias=True),
          "Small (2.6M)", prompt_len=32)

    # 中等模型中等序列
    bench(GPTConfig(vocab_size=1000, block_size=512,
                    n_layer=12, n_head=12, n_embd=384, dropout=0.0, bias=True),
          "Medium (8.6M)", prompt_len=32)

    # 强调场景: 短 prompt, 长输出 → cache 优势明显
    bench(GPTConfig(vocab_size=1000, block_size=1024,
                    n_layer=12, n_head=8, n_embd=512, dropout=0.0, bias=True),
          "Large (22M)", prompt_len=16)

    # CPU 模式对比（凸显 O(n^2) vs O(n) 差异，不需要 GPU）
    print("\n--- CPU 模式 (差异更明显) ---")
    cfg = GPTConfig(vocab_size=1000, block_size=512, n_layer=6, n_head=4,
                    n_embd=192, dropout=0.0, bias=True)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, 1000, (1, 16))
    n_p = model.get_num_params(non_embedding=False) / 1e6
    print(f"  ({n_p:.1f}M params)")

    for N in [20, 50, 100]:
        t0 = time.perf_counter()
        model.generate(prompt, N, use_cache=False)
        t_nocache = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.generate(prompt, N, use_cache=True)
        t_cache = time.perf_counter() - t0
        print(f"  生成 {N:>3d} tokens:  no-cache={t_nocache*1000:6.0f}ms  "
              f"cache={t_cache*1000:6.0f}ms  speedup={t_nocache/t_cache:.1f}x")

    # ── 无 Flash Attention (凸显原始 O(n^2) vs O(n) 差距) ──
    print("\n--- CPU 禁用 Flash Attention (O(n^2) vs O(n) 原始对比) ---")
    cfg2 = GPTConfig(vocab_size=1000, block_size=256, n_layer=4, n_head=4,
                     n_embd=128, dropout=0.0, bias=True)
    model2 = GPT(cfg2).eval()
    T = cfg2.block_size
    # 注意: 直接设置内部属性是脆弱的 —— 依赖 CausalSelfAttention 的实现细节。
    # 若 attention 实现变更（重命名 flash/移除 bias），此段可能静默失效。
    for block in model2.transformer.h:
        try:
            block.attn.flash = False
        except AttributeError:
            print("  [WARN] 无法禁用 Flash Attention（attn 结构已变更），跳过此测试。")
            break
        if not hasattr(block.attn, 'bias'):
            mask = torch.tril(torch.ones(T, T))
            block.attn.register_buffer('bias', mask.view(1, 1, T, T))
    else:
        n_p = model2.get_num_params(non_embedding=False) / 1e6
        prompt2 = torch.randint(0, 1000, (1, 8))
        print(f"  ({n_p:.1f}M params, Flash Attn OFF, O(n^2) attention)")

        for N in [20, 50, 100]:
            t0 = time.perf_counter()
            model2.generate(prompt2, N, use_cache=False)
            t_nocache = time.perf_counter() - t0
            t0 = time.perf_counter()
            model2.generate(prompt2, N, use_cache=True)
            t_cache = time.perf_counter() - t0
            print(f"  生成 {N:>3d} tokens:  no-cache={t_nocache*1000:6.0f}ms  "
                  f"cache={t_cache*1000:6.0f}ms  speedup={t_nocache/t_cache:.1f}x")
