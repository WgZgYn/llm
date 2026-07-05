"""训练入口 —— 组装模型、后端、数据，启动 Trainer。

并行策略（DDP/FSDP/Single）在这里完成 PyTorch 原生包装。
Trainer 不感知具体后端，只运行标准训练循环。

用法:
    python scripts/train.py configs/shakespeare_char.py
    torchrun --nproc-per-node=4 scripts/train.py configs/gpt2_124M_owt.py

配置文件职责:
    必须定义: model (GPTConfig) + training (TrainingConfig)
    可选定义: model_obj (__init__ 完成的模型, 用于变种如 MoE)
              此时 model 仍需要提供 GPTConfig 供校验
"""

import os
import sys
import argparse
import importlib.util

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm import GPT, GPTConfig, TrainingConfig, validate_configs
from llm.data.loader import create_dataloader
from llm.training.trainer import Trainer


# ═══════════════════════════════════════════════════════════════
# 配置加载
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: str):
    """动态加载 Python 配置文件，返回 (model_cfg, train_cfg, module)。"""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    spec = importlib.util.spec_from_file_location("task_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    missing = []
    if not hasattr(module, "model"): missing.append("model (GPTConfig)")
    if not hasattr(module, "training"): missing.append("training (TrainingConfig)")
    if missing:
        raise ValueError(f"配置文件缺少: {missing}")

    if not isinstance(module.model, GPTConfig):
        raise TypeError(f"model 必须是 GPTConfig, 实际 {type(module.model)}")
    if not isinstance(module.training, TrainingConfig):
        raise TypeError(f"training 必须是 TrainingConfig, 实际 {type(module.training)}")
    return module.model, module.training, module


# ═══════════════════════════════════════════════════════════════
# DDP 环境
# ═══════════════════════════════════════════════════════════════

def _init_distributed():
    """初始化 torch.distributed，返回 (rank, local_rank, world_size, device)。"""
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


# ═══════════════════════════════════════════════════════════════
# 模型构造（所有模型逻辑集中在这里）
# ═══════════════════════════════════════════════════════════════

def build_model(model_cfg: GPTConfig, train_cfg: TrainingConfig,
                config_module, device: torch.device,
                is_dist: bool, local_rank: int):
    """创建模型 + 分布式包装 + torch.compile。

    返回:
        model:      包装后的模型（可能是 DDP/FSDP/裸）
        raw_model:  原始模型（checkpoint 保存用）
        ddp:        是否 DDP 包装（Trainer 需要知道以使用 no_sync）
    """
    # ── step 1: 创建原始模型 ──
    if hasattr(config_module, "model_obj"):
        # 配置文件直接提供构造好的模型（变种模型：MoE 等）
        raw_model = config_module.model_obj
        print(f"Using pre-built model: {type(raw_model).__name__}")
    elif train_cfg.init_from not in ("scratch", "resume"):
        # 从 HuggingFace 加载预训练权重
        raw_model = GPT.from_pretrained(train_cfg.init_from, override_args={
            k: v for k, v in {"dropout": model_cfg.dropout}.items()
            if v != GPTConfig().dropout
        })
        if model_cfg.block_size < raw_model.config.block_size:
            raw_model.crop_block_size(model_cfg.block_size)
    else:
        # 从零初始化
        raw_model = GPT(model_cfg)

    # ── step 2: 分布式包装 ──
    backend = train_cfg.backend.lower()
    ddp = False

    if backend == "fsdp" and is_dist:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        ptdtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[train_cfg.dtype]
        mp = MixedPrecisionPolicy(param_dtype=ptdtype, reduce_dtype=ptdtype)

        raw_model = raw_model.to(device)
        for block in raw_model.transformer.h:
            fully_shard(block, mp_policy=mp,
                        reshard_after_forward=train_cfg.fsdp_reshard)
        fully_shard(raw_model, mp_policy=mp,
                    reshard_after_forward=train_cfg.fsdp_reshard)
        model = raw_model  # FSDP 原地包装，raw_model 和 model 是同一个对象

    elif backend == "ddp" and is_dist:
        raw_model = raw_model.to(device)
        model = DDP(raw_model, device_ids=[local_rank])
        ddp = True

    else:
        raw_model = raw_model.to(device)
        model = raw_model

    # ── step 3: torch.compile ──
    if train_cfg.compile and hasattr(torch, 'compile'):
        try:
            import triton  # noqa: F401
        except ImportError:
            print("[WARNING] Triton not available, skipping torch.compile")
        else:
            print("compiling the model... (takes ~a minute)")
            model = torch.compile(model)

    return model, raw_model, ddp


# ═══════════════════════════════════════════════════════════════
# 优化器
# ═══════════════════════════════════════════════════════════════

def build_optimizer(model, config: TrainingConfig):
    """AdamW，2D 参数用 weight_decay，bias/norm 不用。"""
    raw = model.module if hasattr(model, 'module') else model
    params = {pn: p for pn, p in raw.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    nodecay = [p for p in params.values() if p.dim() < 2]
    groups = [
        {'params': decay, 'weight_decay': config.weight_decay},
        {'params': nodecay, 'weight_decay': 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=config.learning_rate,
                             betas=(config.beta1, config.beta2), fused=fused)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT 模型训练")
    parser.add_argument("config", type=str, help="配置文件路径")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--compile", action="store_true", default=None,
                        dest="force_compile")
    args = parser.parse_args()
    return args


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 1. 加载 & 校验配置 ──
    args = parse_args()
    print(f"Loading config: {args.config}")
    model_cfg, train_cfg, config_module = load_config(args.config)
    validate_configs(model_cfg, train_cfg, strict=not args.no_strict)
    print("  Config validation passed")
    if args.force_compile is not None:
        train_cfg.compile = args.force_compile

    # ── 2. 分布式环境 ──
    is_dist = int(os.environ.get('RANK', -1)) != -1
    rank, local_rank, world_size = 0, 0, 1
    device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')

    if is_dist:
        rank, local_rank, world_size, device = _init_distributed()
        if train_cfg.backend.lower() not in ("ddp", "fsdp"):
            print(f"[WARNING] torchrun 下 backend='{train_cfg.backend}'，退化单卡")

    # ── 3. 打印 ──
    if rank == 0:
        n_params = (12 * model_cfg.n_layer * model_cfg.n_embd ** 2) / 1e6
        total_batch = train_cfg.batch_size * train_cfg.gradient_accumulation_steps * world_size
        print(f"\n{'=' * 60}")
        print(f"  Model:     {model_cfg.n_layer}L {model_cfg.n_head}H "
              f"{model_cfg.n_embd}D  (~{n_params:.1f}M params)")
        print(f"  Training:  {train_cfg.max_iters} steps, "
              f"global_batch={total_batch}")
        print(f"             lr={train_cfg.learning_rate}, "
              f"dtype={train_cfg.dtype}, backend={train_cfg.backend}")
        print(f"{'=' * 60}\n")

    # ── 4. 模型（创建 + 分布式包装 + compile，集中在此）──
    model, raw_model, ddp = build_model(
        model_cfg, train_cfg, config_module, device, is_dist, local_rank)

    # ── 5. 优化器 + scaler ──
    optimizer = build_optimizer(model, train_cfg)
    use_fp16 = train_cfg.dtype == 'float16' and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) if use_fp16 else None

    # ── 6. bf16 兼容性检查 ──
    if device.type == 'cuda' and train_cfg.dtype == 'bfloat16':
        if not torch.cuda.is_bf16_supported():
            gpu_name = torch.cuda.get_device_name(device)
            print(f"[WARNING] {gpu_name} 不支持 bfloat16, 自动切换 float16")
            train_cfg.dtype = 'float16'
            scaler = torch.amp.GradScaler('cuda', enabled=True)

    # ── 7. 数据 ──
    data_dir = train_cfg.data_dir or os.path.join(PROJECT_ROOT, "data", train_cfg.dataset)
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        print(f"[ERROR] 未找到训练数据: {data_dir}/train.bin")
        sys.exit(1)

    train_loader = create_dataloader(
        data_dir, "train", train_cfg.batch_size, model_cfg.block_size)
    val_loader = create_dataloader(
        data_dir, "val", train_cfg.batch_size, model_cfg.block_size)

    # ── 8. Trainer ──
    trainer = Trainer(
        model=model, optimizer=optimizer, config=train_cfg,
        train_loader=train_loader, val_loader=val_loader,
        raw_model=raw_model, scaler=scaler, ddp=ddp,
    )

    if train_cfg.init_from == "resume":
        trainer.load_checkpoint()

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
