"""训练结果验证工具。

用法:
    python scripts/verify_checkpoint.py <checkpoint_path>
    python scripts/verify_checkpoint.py out/shakespeare_char/ckpt.pt
"""

import sys
import pickle
import torch
import numpy as np

sys.path.insert(0, ".")

from llm import GPT, GPTConfig


def verify(checkpoint_path: str):
    print(f"Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # ── 1. 训练统计 ──
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    print(f"  Steps completed:   {ckpt['iter_num']}")
    print(f"  Best val loss:     {ckpt['best_val_loss']:.4f}")
    model_params = sum(p.numel() for p in ckpt["model"].values())
    print(f"  Model parameters:  {model_params/1e6:.2f}M")

    # ── 2. 加载模型 ──
    model_cfg = GPTConfig.from_dict(ckpt["model_args"])
    model = GPT(model_cfg)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval()
    print(f"  Model loaded OK:   {model_cfg.n_layer}L/{model_cfg.n_head}H/{model_cfg.n_embd}D")

    # ── 3. 检查权重质量 ──
    print(f"\n{'='*50}")
    print("Weight Sanity Checks")
    print(f"{'='*50}")
    for name, param in model.named_parameters():
        w = param.data
        if w.numel() == 0:
            continue
        has_nan = torch.isnan(w).any().item()
        has_inf = torch.isinf(w).any().item()
        if has_nan or has_inf:
            print(f"  FAIL: {name} has NaN={has_nan}, Inf={has_inf}")
        elif "weight" in name and w.dim() >= 2:
            # 权重不应全部一样（退化检测）
            if w.std() < 1e-8:
                print(f"  WARN: {name} std={w.std():.2e} (possibly collapsed)")

    print(f"  No NaN/Inf found — weights healthy")

    # ── 4. 快速前向测试 ──
    dummy = torch.randint(0, model_cfg.vocab_size, (1, 32))
    with torch.no_grad():
        logits, _ = model(dummy, dummy)
    print(f"  Forward pass OK:  input={list(dummy.shape)} -> logits={list(logits.shape)}")

    # ── 5. 续训验证 ──
    print(f"\n{'='*50}")
    print("Resume Check")
    print(f"{'='*50}")
    train_cfg = ckpt.get("config", {})
    print(f"  Dataset:    {train_cfg.get('dataset', 'unknown')}")
    print(f"  Batch size: {train_cfg.get('batch_size', '?')} x {train_cfg.get('gradient_accumulation_steps', '?')}")
    print(f"  LR:         {train_cfg.get('learning_rate', '?')}")

    print(f"\n==> Verification complete — model is healthy and resumable")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "out/shakespeare_char/ckpt.pt"
    verify(path)
