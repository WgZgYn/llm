"""LoRA 工具函数 —— state_dict 管理、merge/unmerge、参数统计。"""

import os
import torch
import torch.nn as nn

from .linear import LoRALinear


# ═══════════════════════════════════════════════════════════════════
# LoRA State Dict 管理
# ═══════════════════════════════════════════════════════════════════

def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """只返回 LoRA 参数（lora_A, lora_B 的权重）。

    用途: 保存 lightweight adapter（~MB 级别），与 base model 分离。

    返回:
        dict: {name: tensor}，只包含 key 中含有 'lora_' 的参数
    """
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if 'lora_' in name
    }


def save_lora_adapters(model: nn.Module, path: str, lora_config=None):
    """保存 LoRA adapter 到文件。

    只保存 LoRA 参数（~MB 级），不保存 base model weights。

    参数:
        model:       包含 LoRA 的模型
        path:        保存路径（如 'out/lora_gpt2/lora_adapters.pt'）
        lora_config: 可选，同时保存 LoRAConfig 供加载时参考
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    save_dict = {
        'lora_state': lora_state_dict(model),
    }
    if lora_config is not None:
        save_dict['lora_config'] = lora_config.to_dict()

    torch.save(save_dict, path)
    print(f"LoRA adapters saved to {path} "
          f"({_format_size(save_dict['lora_state'])} params)")


def load_lora_adapters(model: nn.Module, path: str) -> dict | None:
    """加载 LoRA adapter 权重。

    参数:
        model: 包含 LoRA 的模型（LoRALinear 已初始化）
        path:  adapter 文件路径

    返回:
        文件中保存的 lora_config dict（如果有），否则 None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA adapter 文件不存在: {path}")

    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    lora_state = checkpoint.get('lora_state', checkpoint)

    # strict=False → 忽略 base model keys，只加载 LoRA keys
    missing, unexpected = model.load_state_dict(lora_state, strict=False)

    # 只报告真正的错误
    lora_keys_in_model = {k for k in model.state_dict() if 'lora_' in k}
    lora_keys_in_file = set(lora_state.keys())
    truly_missing = lora_keys_in_model - lora_keys_in_file
    truly_unexpected = lora_keys_in_file - lora_keys_in_model

    if truly_missing:
        print(f"[WARNING] LoRA keys in model but not in file: {truly_missing}")
    if truly_unexpected:
        print(f"[WARNING] LoRA keys in file but not in model: {truly_unexpected}")

    print(f"LoRA adapters loaded from {path}")
    return checkpoint.get('lora_config', None)


# ═══════════════════════════════════════════════════════════════════
# Merge / Unmerge
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def merge_all_lora(model: nn.Module):
    """融合模型中所有 LoRA 权重到 base weight。

    融合后 forward 等价于普通 nn.Linear，推理零开销。
    之后可以安全地 torch.save 整个模型，且不需要 LoRA 代码来推理。
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            count += 1
    if count > 0:
        print(f"Merged {count} LoRA layers into base weights")
    return count


@torch.no_grad()
def unmerge_all_lora(model: nn.Module):
    """还原所有已融合的 LoRA 权重，恢复可训练状态。"""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            count += 1
    if count > 0:
        print(f"Unmerged {count} LoRA layers from base weights")
    return count


# ═══════════════════════════════════════════════════════════════════
# 参数统计
# ═══════════════════════════════════════════════════════════════════

def count_lora_params(model: nn.Module) -> dict:
    """统计 LoRA 相关参数。

    返回:
        dict with keys:
            total_params:      总参数量
            trainable_params:  可训练参数量
            lora_params:       仅 LoRA 参数量（lora_A + lora_B）
            lora_ratio:        LoRA 参数 / 总参数
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_only = sum(
        p.numel() for n, p in model.named_parameters() if 'lora_' in n
    )
    return {
        'total_params': total,
        'trainable_params': trainable,
        'lora_params': lora_only,
        'lora_ratio': lora_only / total if total > 0 else 0.0,
    }


def print_lora_info(model: nn.Module):
    """打印 LoRA 参数信息。"""
    info = count_lora_params(model)
    print(f"Total parameters:       {info['total_params']/1e6:.2f}M")
    print(f"Trainable parameters:   {info['trainable_params']/1e6:.2f}M")
    print(f"LoRA-only parameters:   {info['lora_params']/1e6:.2f}M "
          f"({info['lora_ratio']*100:.2f}% of total)")

    # 逐层打印 LoRA 参数
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_params = module.lora_A.weight.numel()
            b_params = module.lora_B.weight.numel()
            print(f"  {name}: r={module.r}, "
                  f"A={a_params:,}, B={b_params:,} "
                  f"({a_params + b_params:,} total)")


# ═══════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════

def _format_size(state_dict: dict) -> str:
    """格式化参数量大小。"""
    total = sum(v.numel() for v in state_dict.values())
    if total >= 1e6:
        return f"{total/1e6:.2f}M"
    elif total >= 1e3:
        return f"{total/1e3:.2f}K"
    return str(total)
