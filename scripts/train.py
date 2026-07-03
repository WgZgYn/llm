"""训练入口脚本。

严格要求传递配置文件，所有超参数在配置文件中显式指定。

用法:
    python scripts/train.py configs/shakespeare_char.py
    python scripts/train.py configs/shakespeare_char.py --no-strict   # 跳过字段完整性检查
    python scripts/train.py configs/shakespeare_char.py --compile     # 强制启用 torch.compile
"""

import os
import sys
import argparse
import importlib.util

# 将项目根目录加入 Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm import GPT, GPTConfig, TrainingConfig, Trainer, create_backend, validate_configs
from llm.data import MemMapDataProvider


def load_config(config_path: str) -> tuple[GPTConfig, TrainingConfig]:
    """从 Python 文件加载配置。

    配置文件必须定义两个变量:
        model:   GPTConfig 实例
        training: TrainingConfig 实例

    返回:
        (model_config, training_config)
    """
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 动态加载 Python 模块
    spec = importlib.util.spec_from_file_location("task_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 检查必需变量
    missing = []
    if not hasattr(module, "model"):
        missing.append("model (GPTConfig)")
    if not hasattr(module, "training"):
        missing.append("training (TrainingConfig)")
    if missing:
        raise ValueError(
            f"配置文件 {config_path} 缺少以下必需变量:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + f"\n\n配置文件格式:\n"
            + "  from llm.config import GPTConfig, TrainingConfig\n\n"
            + "  model = GPTConfig(vocab_size=..., block_size=..., ...)\n"
            + "  training = TrainingConfig(dataset=..., max_iters=..., ...)\n"
        )

    model_cfg = module.model
    train_cfg = module.training

    if not isinstance(model_cfg, GPTConfig):
        raise TypeError(f"model 必须是 GPTConfig 实例，实际为 {type(model_cfg)}")
    if not isinstance(train_cfg, TrainingConfig):
        raise TypeError(f"training 必须是 TrainingConfig 实例，实际为 {type(train_cfg)}")

    return model_cfg, train_cfg


def main():
    parser = argparse.ArgumentParser(
        description="GPT 模型训练",
        usage="python scripts/train.py <config_file> [--no-strict] [--compile]",
    )
    parser.add_argument(
        "config",
        type=str,
        help="训练配置文件路径（Python 文件，必需）",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="跳过配置字段完整性检查（不推荐）",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=None,
        dest="force_compile",
        help="强制启用 torch.compile（覆盖配置文件中的设置）",
    )

    args = parser.parse_args()

    # ── 1. 加载配置 ──
    print(f"Loading config: {args.config}")
    model_cfg, train_cfg = load_config(args.config)

    # ── 2. 校验配置完整性 ──
    print("Validating config completeness...")
    validate_configs(model_cfg, train_cfg, strict=not args.no_strict)
    print("  Config validation passed")

    # ── 3. 覆盖 compile（可选）──
    if args.force_compile is not None:
        train_cfg.compile = args.force_compile

    # ── 4. 打印关键信息 ──
    print(f"\n{'=' * 60}")
    print(f"Model:      {model_cfg.n_layer} layers, {model_cfg.n_head} heads, "
          f"{model_cfg.n_embd} dim")
    param_est = (12 * model_cfg.n_layer * model_cfg.n_embd ** 2) / 1e6
    print(f"            ~{param_est:.1f}M params (estimate)")
    print(f"Training:   {train_cfg.max_iters} steps, "
          f"batch={train_cfg.batch_size}×{train_cfg.gradient_accumulation_steps}")
    print(f"            lr={train_cfg.learning_rate}, "
          f"dtype={train_cfg.dtype}, backend={train_cfg.backend}")
    print(f"{'=' * 60}\n")

    # ── 5. 创建模型 ──
    if train_cfg.init_from not in ("scratch", "resume"):
        # 从 HuggingFace GPT-2 加载
        model = GPT.from_pretrained(train_cfg.init_from, override_args={
            k: v for k, v in {"dropout": model_cfg.dropout}.items()
            if v != GPTConfig().dropout  # 只覆盖非默认值
        })
        if model_cfg.block_size < model.config.block_size:
            model.crop_block_size(model_cfg.block_size)
    else:
        model = GPT(model_cfg)

    # ── 6. 创建后端 ──
    backend = create_backend(train_cfg)
    print(f"Backend: {type(backend).__name__}, device: {backend.device}")

    # ── 7. 创建数据 ──
    data_dir = train_cfg.data_dir or os.path.join(PROJECT_ROOT, "data", train_cfg.dataset)
    if os.path.exists(os.path.join(data_dir, "train.bin")):
        data = MemMapDataProvider(data_dir, train_cfg.batch_size, model_cfg.block_size)
    else:
        print(f"[ERROR] 未找到训练数据: {data_dir}/train.bin")
        print(f"请先运行数据预处理脚本，如: python llm/data/prepare/{train_cfg.dataset}.py")
        sys.exit(1)

    # ── 8. 创建 Trainer 并启动 ──
    trainer = Trainer(model, train_cfg, backend, data_provider=data)

    if train_cfg.init_from == "resume":
        trainer.load_checkpoint()

    trainer.train()


if __name__ == "__main__":
    main()
