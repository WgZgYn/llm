"""训练后端抽象 —— 解耦分布式策略与训练循环。

提供:
    TrainingBackend (ABC):  抽象接口
    SingleDeviceBackend:    单卡/CPU
    DDPBackend:             DistributedDataParallel
    DeepSpeedBackend:       DeepSpeed ZeRO (stub, 待实现)
    create_backend():       工厂函数，从 TrainingConfig 自动选择
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import ContextManager
import inspect

import torch
import torch.nn as nn

from llm.config.training_config import TrainingConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 抽象基类
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingBackend(ABC):
    """分布式训练后端的抽象接口。

    设计目标: 让 Trainer 的循环逻辑与分布式策略完全解耦。
    Trainer 不直接调用 loss.backward() / optimizer.step() / model.to(device)，
    而是通过此后端接口，由具体后端处理设备放置、混合精度、梯度同步等。

    标准训练步调用顺序（由 Trainer 编排）:
        1. backend.zero_grad(optimizer)
        2. for micro_step in range(grad_accum_steps):
        3.     with backend.autocast_context():        ← 混合精度前向
        4.         logits, loss = model(X, Y)
        5.     backend.backward(model, loss, is_last)   ← 反向 + 梯度同步
        6. backend.unscale(optimizer)                   ← fp16 时 unscale
        7. clip_grad_norm_(model.parameters(), ...)     ← 梯度裁剪（Trainer 直接做）
        8. backend.step(optimizer)                      ← 优化器更新 + scaler.update
    """

    # ── 属性 ──

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """当前进程使用的设备。"""
        ...

    @property
    @abstractmethod
    def world_size(self) -> int:
        """总进程数（单卡=1）。"""
        ...

    @property
    @abstractmethod
    def rank(self) -> int:
        """当前进程的全局排名。"""
        ...

    @property
    @abstractmethod
    def is_master(self) -> bool:
        """当前进程是否为主进程（负责日志和 checkpoint）。"""
        ...

    # ── 生命周期 ──

    @abstractmethod
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """模型设备放置 + 分布式包装。返回用于前向传播的模型对象。"""
        ...

    @abstractmethod
    def prepare_optimizer(
        self, model: nn.Module, config: TrainingConfig
    ) -> torch.optim.Optimizer:
        """创建优化器（DeepSpeed 内部管理，Single/DDP 使用标准 AdamW）。"""
        ...

    # ── 训练步操作 ──

    @abstractmethod
    def autocast_context(self) -> ContextManager:
        """混合精度前向传播的上下文管理器。"""
        ...

    @abstractmethod
    def backward(self, model: nn.Module, loss: torch.Tensor,
                 is_last_micro_step: bool = True):
        """反向传播。

        DDP: 非最后 micro_step 使用 model.no_sync() 跳过 all-reduce。
        fp16: 通过 GradScaler 缩放 loss。
        """
        ...

    @abstractmethod
    def unscale(self, optimizer: torch.optim.Optimizer):
        """fp16: 将梯度从 scaler 的缩放中还原（bf16/fp32 下为 no-op）。
        必须在梯度裁剪之前调用。"""
        ...

    @abstractmethod
    def step(self, optimizer: torch.optim.Optimizer):
        """优化器一步更新 + GradScaler.update()。"""
        ...

    @abstractmethod
    def zero_grad(self, optimizer: torch.optim.Optimizer):
        """清零所有梯度。"""
        ...

    def cleanup(self):
        """训练结束清理（DDP 需 destroy_process_group）。"""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 分组 Weight Decay 优化器（Single 和 DDP 共用）
# ═══════════════════════════════════════════════════════════════════════════════

def _create_optimizer(
    model: nn.Module, config: TrainingConfig, device_type: str
) -> torch.optim.Optimizer:
    """创建 AdamW 优化器，分组 weight decay。

    AdamW 标准做法: 2D 参数（权重矩阵）→ decay；<2D（bias/norm）→ 不 decay。
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'

    return torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=use_fused,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SingleDeviceBackend
# ═══════════════════════════════════════════════════════════════════════════════

class SingleDeviceBackend(TrainingBackend):
    """单卡/CPU 训练后端。"""

    def __init__(self, config: TrainingConfig):
        device_str = config.device
        if device_str == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device_str = "cpu"

        self._device = torch.device(device_str)
        self._world_size = 1
        self._rank = 0

        self.dtype = config.dtype
        self.device_type = 'cuda' if self._device.type == 'cuda' else 'cpu'
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[self.dtype]

        # GradScaler: 仅 fp16 需要，bf16 范围大不需要
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == 'float16'))

        # TF32 加速（CUDA）
        if self.device_type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ── 属性 ──
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_master(self) -> bool:
        return True

    # ── 生命周期 ──
    def prepare_model(self, model: nn.Module) -> nn.Module:
        return model.to(self._device)

    def prepare_optimizer(
        self, model: nn.Module, config: TrainingConfig
    ) -> torch.optim.Optimizer:
        return _create_optimizer(model, config, self.device_type)

    # ── 训练步操作 ──
    def autocast_context(self) -> ContextManager:
        if self.device_type == 'cpu':
            return nullcontext()
        return torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

    def backward(self, model: nn.Module, loss: torch.Tensor,
                 is_last_micro_step: bool = True):
        self.scaler.scale(loss).backward()

    def unscale(self, optimizer: torch.optim.Optimizer):
        """fp16 下还原梯度缩放，bf16/fp32 下 no-op。"""
        if self.dtype == 'float16':
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()

    def zero_grad(self, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad(set_to_none=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DDPBackend
# ═══════════════════════════════════════════════════════════════════════════════

class DDPBackend(TrainingBackend):
    """DistributedDataParallel 训练后端。

    关键设计:
    - 梯度累积时，非最后 micro_step 通过 model.no_sync() 跳过 all-reduce
    - Ring AllReduce 在所有 GPU 间平等同步梯度
    """

    def __init__(self, config: TrainingConfig):
        import os
        import torch.distributed as dist

        # Windows: PyTorch 子进程默认 use_libuv=True 但 Windows wheel 不带 libuv
        os.environ["USE_LIBUV"] = "0"

        dist.init_process_group(backend=config.ddp_backend)

        self._rank = int(os.environ['RANK'])
        self._local_rank = int(os.environ['LOCAL_RANK'])
        self._world_size = int(os.environ['WORLD_SIZE'])

        self._device = torch.device(f'cuda:{self._local_rank}')
        torch.cuda.set_device(self._device)

        self.dtype = config.dtype
        self.device_type = 'cuda'
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[self.dtype]

        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == 'float16'))

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 属性 ──
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_master(self) -> bool:
        return self._rank == 0

    # ── 生命周期 ──
    def prepare_model(self, model: nn.Module) -> nn.Module:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = model.to(self._device)
        model = DDP(model, device_ids=[self._local_rank])
        return model

    def prepare_optimizer(
        self, model: nn.Module, config: TrainingConfig
    ) -> torch.optim.Optimizer:
        raw_model = model.module if hasattr(model, 'module') else model
        return _create_optimizer(raw_model, config, self.device_type)

    # ── 训练步操作 ──
    def autocast_context(self) -> ContextManager:
        return torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

    def backward(self, model: nn.Module, loss: torch.Tensor,
                 is_last_micro_step: bool = True):
        """DDP 反向传播 + 梯度同步控制。

        关键: 非最后 micro_step 使用 model.no_sync() 跳过 all-reduce，
        避免在每个 micro_step 都做一次全局通信。
        """
        if is_last_micro_step:
            self.scaler.scale(loss).backward()
        else:
            # no_sync() 告诉 DDP 本次 backward 不同步梯度
            with model.no_sync():
                self.scaler.scale(loss).backward()

    def unscale(self, optimizer: torch.optim.Optimizer):
        if self.dtype == 'float16':
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()

    def zero_grad(self, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad(set_to_none=True)

    def cleanup(self):
        import torch.distributed as dist
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════════
# FSDPBackend
# ═══════════════════════════════════════════════════════════════════════════════

class FSDPBackend(TrainingBackend):
    """Fully Sharded Data Parallel 训练后端（PyTorch ≥ 2.4）。

    与 DDP 的核心区别:
    - DDP: 每 GPU 保留完整模型副本，仅同步梯度（all-reduce）
    - FSDP: 参数/梯度/优化器状态分片到多 GPU，前向/反向时按需重组

    分片策略（通过 fsdp_reshard 控制）:
    - reshard=True  (FULL_SHARD):  每层前向后立即释放参数 → 类 ZeRO-3，最省显存
    - reshard=False (SHARD_GRAD_OP): 保留参数，仅分片梯度 → 类 ZeRO-2，更快但更多显存

    FSDP 通过 autograd hooks 管理参数重组/释放，
    backward/step/zero_grad 与单卡完全一致，无需 no_sync 等特殊处理。
    """

    def __init__(self, config: TrainingConfig):
        import os
        import torch.distributed as dist
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        os.environ["USE_LIBUV"] = "0"
        dist.init_process_group(backend=config.ddp_backend)

        self._rank = int(os.environ['RANK'])
        self._local_rank = int(os.environ['LOCAL_RANK'])
        self._world_size = int(os.environ['WORLD_SIZE'])

        self._device = torch.device(f'cuda:{self._local_rank}')
        torch.cuda.set_device(self._device)

        self.dtype = config.dtype
        self.device_type = 'cuda'
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[self.dtype]

        self.fsdp_reshard = config.fsdp_reshard

        # FSDP 混合精度: 与 autocast 独立，控制参数存储精度
        self.mp_policy = MixedPrecisionPolicy(
            param_dtype=self.ptdtype,
            reduce_dtype=self.ptdtype,
        )

        # GradScaler（仅 fp16 需要）
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == 'float16'))

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 属性 ──
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_master(self) -> bool:
        return self._rank == 0

    # ── 生命周期 ──
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """FSDP2 包装：逐 block 分片，最后包裹整体。

        分片粒度 = 每个 TransformerBlock 一个 FSDP unit:
        - 每个 block 的前向/反向独立管理参数重组
        - 比整个模型包在一起更高效（更细粒度的 prefetch）

        顺序很重要：先分片内部 block，再分片外层模型。
        """
        from torch.distributed.fsdp import fully_shard

        model = model.to(self._device)

        # 逐 block 分片 —— 每个 Block 成为独立的 FSDP unit
        for block in model.transformer.h:
            fully_shard(
                block,
                mp_policy=self.mp_policy,
                reshard_after_forward=self.fsdp_reshard,
            )

        # 最后分片整个模型（wte, wpe, ln_f, lm_head 等外层参数）
        fully_shard(
            model,
            mp_policy=self.mp_policy,
            reshard_after_forward=self.fsdp_reshard,
        )

        return model

    def prepare_optimizer(
        self, model: nn.Module, config: TrainingConfig
    ) -> torch.optim.Optimizer:
        # FSDP 包装后 model.parameters() 返回的是 FSDP 管理的 FlatParameter，
        # _create_optimizer 可以直接使用（分片后的参数）
        return _create_optimizer(model, config, self.device_type)

    # ── 训练步操作 ──
    def autocast_context(self) -> ContextManager:
        return torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

    def backward(self, model: nn.Module, loss: torch.Tensor,
                 is_last_micro_step: bool = True):
        """FSDP 反向传播。

        FSDP 通过 autograd hooks 自动管理参数重组/释放，
        backward 调用与单卡完全一致。
        梯度累积优化: FSDP 不需要 no_sync ——
        reduce-scatter 在每个 backward 中独立完成，grad 自然累积。
        """
        self.scaler.scale(loss).backward()

    def unscale(self, optimizer: torch.optim.Optimizer):
        if self.dtype == 'float16':
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()

    def zero_grad(self, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad(set_to_none=True)

    def cleanup(self):
        import torch.distributed as dist
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSpeedBackend (stub)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSpeedBackend(TrainingBackend):
    """DeepSpeed ZeRO 训练后端 —— 未来实现的桩。

    待实现时需:
    1. pip install deepspeed
    2. 编写 DeepSpeed config JSON
    3. deepspeed.initialize(model, config=ds_config)
    4. engine.backward / engine.step 替代手动管理
    """

    def __init__(self, config: TrainingConfig):
        raise NotImplementedError(
            "DeepSpeed backend not yet implemented.\n"
            "Install: pip install deepspeed\n"
            "Reference: https://www.deepspeed.ai/getting-started/"
        )

    @property
    def device(self): ...
    @property
    def world_size(self): ...
    @property
    def rank(self): ...
    @property
    def is_master(self): ...
    def prepare_model(self, model): ...
    def prepare_optimizer(self, model, config): ...
    def autocast_context(self): ...
    def backward(self, model, loss, is_last_micro_step=True): ...
    def unscale(self, optimizer): ...
    def step(self, optimizer): ...
    def zero_grad(self, optimizer): ...


# ═══════════════════════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def create_backend(config: TrainingConfig) -> TrainingBackend:
    """从配置自动选择并创建训练后端。"""
    backend_type = config.backend.lower()

    if backend_type == "single":
        return SingleDeviceBackend(config)

    elif backend_type == "ddp":
        import os
        is_ddp = int(os.environ.get('RANK', -1)) != -1
        if is_ddp:
            return DDPBackend(config)
        else:
            print("[WARNING] backend='ddp' but no torchrun env detected, "
                  "falling back to single device")
            return SingleDeviceBackend(config)

    elif backend_type == "fsdp":
        import os
        is_fsdp = int(os.environ.get('RANK', -1)) != -1
        if is_fsdp:
            return FSDPBackend(config)
        else:
            print("[WARNING] backend='fsdp' but no torchrun env detected, "
                  "falling back to single device")
            return SingleDeviceBackend(config)

    elif backend_type == "deepspeed":
        return DeepSpeedBackend(config)

    else:
        raise ValueError(
            f"Unknown backend type: '{backend_type}'. "
            f"Choose: single, ddp, fsdp, deepspeed"
        )
