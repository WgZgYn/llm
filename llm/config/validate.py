"""配置文件校验工具。

强制用户显式设置关键超参数，避免静默依赖默认值。
best-practice 类字段（如 dropout=0.0, beta1=0.9）允许与默认值一致。
"""

from dataclasses import fields
from ..config.model_config import GPTConfig
from ..config.training_config import TrainingConfig

# ── 必须显式设置的字段 ──
# 默认值因任务而异，不显式设置可能带来错误行为

MODEL_CRITICAL = {
    "vocab_size", "block_size", "n_layer", "n_head", "n_embd",
}

TRAIN_CRITICAL = {
    "data_dir", "out_dir",
    "max_iters", "batch_size", "gradient_accumulation_steps",
    "learning_rate", "lr_decay_iters", "warmup_iters", "min_lr",
    "eval_interval", "eval_iters",
}


def _is_unset(val) -> bool:
    """检测值是否为其类型的"未设置"标记。"""
    if val is None:
        return True
    if isinstance(val, str) and val == "":
        return True
    if isinstance(val, (int, float)) and val == 0:
        return True
    return False


def check_explicit(config, critical: set[str]) -> list[str]:
    """找出关键字段中未显式设置的（0 / \"\" / None）。"""
    untouched = []
    for name in critical:
        if _is_unset(getattr(config, name)):
            untouched.append(name)
    return untouched


def validate_configs(
    model: GPTConfig,
    training: TrainingConfig,
    strict: bool = True,
) -> bool:
    """校验配置完整性。

    strict=True: 关键字段未显式设置时报错
    strict=False: 仅打印警告
    """
    model_issues = check_explicit(model, MODEL_CRITICAL)
    train_issues = check_explicit(training, TRAIN_CRITICAL)

    messages = []
    if model_issues:
        messages.append(
            f"GPTConfig 以下关键字段未在配置文件中显式设置（当前使用默认值）:\n"
            + "\n".join(f"  - {n} = {getattr(model, n)!r}" for n in model_issues)
        )
    if train_issues:
        messages.append(
            f"TrainingConfig 以下关键字段未在配置文件中显式设置（当前使用默认值）:\n"
            + "\n".join(f"  - {n} = {getattr(training, n)!r}" for n in train_issues)
        )

    if messages:
        msg = "\n\n".join(messages)
        hint = (
            "\n\n这些字段的默认值可能不适用于当前任务，"
            "请在配置文件中显式指定它们。\n"
            "如确认默认值可用，加 --no-strict 跳过此检查。"
        )
        if strict:
            raise ValueError(
                f"配置完整性检查失败 —— 以下关键字段未显式设置:\n\n{msg}{hint}"
            )
        else:
            print(f"[WARNING] {msg}")

    return len(messages) == 0
