"""模型参数分析 —— 参数量、组成、训练显存估算。

用法:
    # 从 config 分析
    python scripts/model_info.py configs/addition_small.py

    # 从 checkpoint 分析
    python scripts/model_info.py out/addition_small/ckpt.pt

    # 从预设分析
    python scripts/model_info.py --preset gpt2
    python scripts/model_info.py --preset gpt2-medium
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import defaultdict

from llm.config.model_config import GPTConfig
from llm.model.gpt import GPT


def analyze(model: torch.nn.Module, batch_size=1, block_size=None,
            dtype="float16", world_size=1, fsdp_stage=0):
    """分析模型参数构成和训练显存需求。"""

    # ── 1. 参数统计 ──
    total = 0
    by_component = defaultdict(int)
    by_shape = defaultdict(list)

    for name, p in model.named_parameters():
        n = p.numel()
        total += n

        # 组件分类
        if "wte" in name:
            by_component["token_embed"] += n
        elif "wpe" in name:
            by_component["pos_embed"] += n
        elif "attn" in name and "weight" in name:
            by_component["attention_qkv" if "c_attn" in name else "attention_proj"] += n
        elif "attn" in name:
            by_component["attention_bias"] += n
        elif "mlp" in name and "weight" in name:
            by_component["mlp_fc" if "c_fc" in name else "mlp_proj"] += n
        elif "mlp" in name:
            by_component["mlp_bias"] += n
        elif "ln" in name:
            by_component["layernorm"] += n
        elif "lm_head" in name:
            by_component["lm_head"] += n
        elif "router" in name:
            by_component["moe_router"] += n
        elif "expert" in name:
            by_component["moe_expert"] += n
        else:
            by_component["other"] += n

        by_shape[str(tuple(p.shape))].append(name)

    # ── 2. 训练显存估算 ──
    bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2}
    param_bytes = bytes_per_param.get(dtype, 2)
    grad_bytes = 4   # 梯度总是 fp32
    opt_bytes = 8    # AdamW 的 m+v = 2×fp32

    mem_params = total * param_bytes
    mem_grads = total * grad_bytes
    mem_optimizer = total * opt_bytes

    if fsdp_stage == 0:     # DDP / single
        mem_static = (mem_params + mem_grads + mem_optimizer) / world_size
    elif fsdp_stage == 2:   # ZeRO-2: 梯度+优化器分片
        mem_static = mem_params + (mem_grads + mem_optimizer) / world_size
    elif fsdp_stage == 3:   # ZeRO-3: 全分片
        mem_static = (mem_params + mem_grads + mem_optimizer) / world_size

    # 激活值粗略估算
    if block_size is None:
        block_size = model.config.block_size if hasattr(model, 'config') else 1024
    n_layers = model.config.n_layer if hasattr(model, 'config') else 12
    n_embd = model.config.n_embd if hasattr(model, 'config') else 768
    # 每层激活 ≈ 34 × B × T × n_embd × 2 bytes (估算)
    mem_activations = n_layers * 34 * batch_size * block_size * n_embd * param_bytes
    if fsdp_stage in (0, 2):  # 非分片参数，激活全在
        pass
    else:  # fsdp_stage=3，激活也受益于分片
        mem_activations /= world_size

    by_component = dict(sorted(by_component.items(), key=lambda x: -x[1]))
    component_pct = {k: 100*v/total for k, v in by_component.items() if v/total > 0.001}

    # ── Kv cache 估算（每个 token）
    n_head = model.config.n_head if hasattr(model, 'config') else 12
    head_dim = n_embd // n_head
    kv_per_token = 2 * n_layers * n_head * head_dim * 2  # bytes
    kv_for_seq = kv_per_token * block_size / 1024**3

    # ── 输出 ──
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Model Analysis")
    lines.append(f"{'='*60}")
    lines.append(f"  Total params:    {total:>12,}  ({total/1e6:.2f}M)")
    lines.append(f"")

    # 按组件
    lines.append(f"  By component:")
    width = max(len(k) for k in by_component) + 2
    for comp, n in by_component.items():
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        lines.append(f"    {comp:<{width}s} {n:>10,} ({pct:5.1f}%)  {bar}")
    lines.append(f"")

    # 按形状
    lines.append(f"  By shape (top 10):")
    for shape, names in sorted(by_shape.items(), key=lambda x: -sum(
            model.get_parameter(n).numel() for n in x[1]))[:10]:
        n_params = sum(model.get_parameter(n).numel() for n in names)
        lines.append(f"    {shape:<24s} x {len(names):>2d} = {n_params:>10,} params")

    lines.append(f"")
    lines.append(f"{'='*60}")
    lines.append(f"Training Memory Estimate (batch={batch_size}, seq={block_size})")
    lines.append(f"{'='*60}")

    def gb(b): return f"{b/1024**3:.2f} GiB"
    lines.append(f"  Parameters:          {gb(mem_params)}")
    lines.append(f"  Gradients:           {gb(mem_grads)}")
    lines.append(f"  Optimizer (AdamW):   {gb(mem_optimizer)}")
    lines.append(f"  Static total:        {gb(mem_params+mem_grads+mem_optimizer)}")
    lines.append(f"")
    lines.append(f"  With FSDP/ZeRO-0 (DDP):   {gb(mem_static)} per GPU")
    if world_size > 1:
        lines.append(f"  With FSDP/ZeRO-2:         {gb(mem_params + (mem_grads+mem_optimizer)/world_size)} per GPU")
        lines.append(f"  With FSDP/ZeRO-3:         {gb((mem_params+mem_grads+mem_optimizer)/world_size)} per GPU")
    lines.append(f"")
    lines.append(f"  Activations (est):   {gb(mem_activations)}")
    lines.append(f"    ( {n_layers}L × 34 × {batch_size} × {block_size} × {n_embd} × {param_bytes}B )")
    lines.append(f"  Peak total (est):    {gb(mem_static + mem_activations)}")
    lines.append(f"")
    lines.append(f"{'='*60}")
    lines.append(f"Inference / KV-Cache")
    lines.append(f"{'='*60}")
    lines.append(f"  KV-Cache per token:  {kv_per_token/1024:.1f} KiB")
    lines.append(f"  KV-Cache for {block_size} tokens: {kv_for_seq:.2f} GiB")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="模型参数分析")
    parser.add_argument("source", nargs="?", help="config 文件或 checkpoint 路径")
    parser.add_argument("--preset", help="预设模型: gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    parser.add_argument("--batch", type=int, default=1, help="显存估算用的 batch size")
    parser.add_argument("--gpus", type=int, default=1, help="GPU 数量")
    parser.add_argument("--fsdp", type=int, default=0, choices=[0,2,3], help="FSDP/ZeRO stage")
    args = parser.parse_args()

    # ── 获取模型 ──
    if args.preset:
        cfg = GPTConfig.from_preset(args.preset, vocab_size=50304, block_size=1024)
        model = GPT(cfg)
    elif args.source and args.source.endswith(".pt"):
        ckpt = torch.load(args.source, map_location="cpu")
        cfg_dict = ckpt.get("model_args", ckpt.get("config", {}))
        if "block_size" not in cfg_dict and "n_layer" in cfg_dict:
            # training config dict
            pass
        cfg = GPTConfig.from_dict(cfg_dict)
        model = GPT(cfg)
        sd = ckpt.get("model", ckpt)
        for k in list(sd.keys()):
            if k.startswith("_orig_mod."):
                sd[k[len("_orig_mod."):]] = sd.pop(k)
        model.load_state_dict(sd, strict=False)
    elif args.source and args.source.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cfg", os.path.abspath(args.source))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        if hasattr(m, "model_obj"):
            model = m.model_obj
            block_size = model.config.block_size
        else:
            cfg = m.model
            model = GPT(cfg)
    else:
        print("Usage: model_info.py <config.py|ckpt.pt|--preset gpt2>")
        return

    # ── 确定 block_size ──
    block_size = getattr(model.config, "block_size", 0) or 1024

    print(analyze(model, batch_size=args.batch, block_size=block_size,
                  world_size=args.gpus, fsdp_stage=args.fsdp))


if __name__ == "__main__":
    main()
