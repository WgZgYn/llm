"""训练入口 —— 组装模型、后端、数据，启动 Trainer。

这是所有并行策略的组装点（DDP / FSDP / DeepSpeed / Single）。
Trainer 不感知具体后端，只运行标准训练循环。

用法:
    python scripts/train.py configs/shakespeare_char.py
    torchrun --nproc-per-node=4 scripts/train.py configs/gpt2_124M_owt.py
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
        raise ValueError(
            f"配置文件缺少: {missing}\n"
            "  model = GPTConfig(...)\n"
            "  training = TrainingConfig(...)"
        )

    model_cfg = module.model
    train_cfg = module.training
    if not isinstance(model_cfg, GPTConfig):
        raise TypeError(f"model 必须是 GPTConfig")
    if not isinstance(train_cfg, TrainingConfig):
        raise TypeError(f"training 必须是 TrainingConfig")
    return model_cfg, train_cfg, module


# ═══════════════════════════════════════════════════════════════
# 并行后端组装
# ═══════════════════════════════════════════════════════════════

def _setup_ddp_env():
    """初始化 DDP process group，返回 (rank, local_rank, world_size)。"""
    # Windows PyTorch 默认 use_libuv=True 但 Windows wheel 不带 libuv
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def _build_model(model_cfg, train_cfg, config_module):
    """创建模型（支持 config 直接传入的 model_obj）。"""
    if hasattr(config_module, "model_obj"):
        model = config_module.model_obj
        print(f"Using pre-built model: {type(model).__name__}")
    elif train_cfg.init_from not in ("scratch", "resume"):
        model = GPT.from_pretrained(train_cfg.init_from, override_args={
            k: v for k, v in {"dropout": model_cfg.dropout}.items()
            if v != GPTConfig().dropout
        })
        if model_cfg.block_size < model.config.block_size:
            model.crop_block_size(model_cfg.block_size)
    else:
        model = GPT(model_cfg)
    return model


def _build_optimizer(model, config: TrainingConfig):
    """AdamW with grouped weight decay（2D 参数 decay，bias/norm 不 decay）。"""
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


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPT 模型训练")
    parser.add_argument("config", type=str, help="配置文件路径")
    parser.add_argument("--no-strict", action="store_true",
                        help="跳过配置完整性检查")
    parser.add_argument("--compile", action="store_true", default=None,
                        dest="force_compile", help="强制启用 torch.compile")
    args = parser.parse_args()

    # ── 1. 加载配置 ──
    print(f"Loading config: {args.config}")
    model_cfg, train_cfg, config_module = load_config(args.config)

    # ── 2. 校验 ──
    print("Validating config completeness...")
    validate_configs(model_cfg, train_cfg, strict=not args.no_strict)
    print("  Config validation passed")
    if args.force_compile is not None:
        train_cfg.compile = args.force_compile

    # ── 3. DDP 环境检测 ──
    is_ddp_env = int(os.environ.get('RANK', -1)) != -1
    backend = train_cfg.backend.lower()
    ddp = False
    rank, local_rank, world_size = 0, 0, 1
    device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')

    if is_ddp_env:
        rank, local_rank, world_size, device = _setup_ddp_env()
        if backend in ("ddp", "fsdp"):
            ddp = (backend == "ddp")
            print(f"[rank{rank}] Initialized {backend}, "
                  f"device={device}, world_size={world_size}")
        else:
            print(f"[WARNING] torchrun 检测到但 backend='{backend}'，"
                  f"将作为单卡运行")
            is_ddp_env = False

    # ── 4. 打印信息 ──
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Model:      {model_cfg.n_layer} layers, {model_cfg.n_head} heads, "
              f"{model_cfg.n_embd} dim")
        param_est = (12 * model_cfg.n_layer * model_cfg.n_embd ** 2) / 1e6
        print(f"            ~{param_est:.1f}M params (estimate)")
        total_batch = train_cfg.batch_size * train_cfg.gradient_accumulation_steps * world_size
        print(f"Training:   {train_cfg.max_iters} steps, total_batch={total_batch}")
        print(f"            lr={train_cfg.learning_rate}, "
              f"dtype={train_cfg.dtype}, backend={backend}")
        print(f"{'=' * 60}\n")

    # ── 5. 模型 ──
    raw_model = _build_model(model_cfg, train_cfg, config_module)

    if backend == "fsdp" and is_ddp_env:
        # FSDP: 逐 block 分片（类 ZeRO-3）
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(
            param_dtype={'float16': torch.float16, 'bfloat16': torch.bfloat16}[train_cfg.dtype],
            reduce_dtype={'float16': torch.float16, 'bfloat16': torch.bfloat16}[train_cfg.dtype],
        )
        raw_model = raw_model.to(device)
        for block in raw_model.transformer.h:
            fully_shard(block, mp_policy=mp, reshard_after_forward=train_cfg.fsdp_reshard)
        fully_shard(raw_model, mp_policy=mp, reshard_after_forward=train_cfg.fsdp_reshard)
        model = raw_model
        ddp = False  # FSDP 不需要 no_sync
    elif ddp:
        raw_model = raw_model.to(device)
        model = DDP(raw_model, device_ids=[local_rank])
    else:
        raw_model = raw_model.to(device)
        model = raw_model

    # ── 6. torch.compile ──
    if train_cfg.compile and hasattr(torch, 'compile'):
        try:
            import triton  # noqa: F401
        except ImportError:
            print("[WARNING] Triton not available, skipping torch.compile")
        else:
            print("compiling the model... (takes ~a minute)")
            model = torch.compile(model)

    # ── 7. 优化器 + scaler ──
    optimizer = _build_optimizer(model, train_cfg)
    use_fp16 = train_cfg.dtype == 'float16' and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) if use_fp16 else None

    # ── 8. 检测 GPU，自动设置 dtype ──
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device)
        if train_cfg.dtype == 'bfloat16':
            # 检查 GPU 是否支持 bf16
            if not torch.cuda.is_bf16_supported():
                print(f"[WARNING] {gpu_name} 不支持 bfloat16，"
                      f"自动切换为 float16")
                train_cfg.dtype = 'float16'
                scaler = torch.amp.GradScaler('cuda', enabled=True)

    # ── 9. 数据 ──
    data_dir = train_cfg.data_dir or os.path.join(
        PROJECT_ROOT, "data", train_cfg.dataset)
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        print(f"[ERROR] 未找到训练数据: {data_dir}/train.bin")
        sys.exit(1)

    train_loader = create_dataloader(
        data_dir, "train", train_cfg.batch_size, model_cfg.block_size)
    val_loader = create_dataloader(
        data_dir, "val", train_cfg.batch_size, model_cfg.block_size)

    # ── 10. Trainer ──
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        raw_model=raw_model,
        scaler=scaler,
        ddp=ddp,
    )

    if train_cfg.init_from == "resume":
        trainer.load_checkpoint()

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
