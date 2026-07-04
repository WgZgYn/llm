"""Trainer v2 —— 直接用 PyTorch DDP/FSDP，不自己抽象 backend。

设计原则:
    - DDP/FSDP/DeepSpeed 是 PyTorch 的事，Trainer 只管训练循环
    - 不再有 TrainingBackend ABC
    - model, optimizer, scaler 由外部准备好后传入

用法:
    # 单卡
    model = GPT(config).to(device)
    trainer = Trainer(model, optimizer, config, loaders)

    # DDP
    model = DDP(GPT(config).to(local_rank), device_ids=[local_rank])
    trainer = Trainer(model, optimizer, config, loaders, ddp=True)

    # FSDP
    model = GPT(config)
    for block in model.transformer.h:
        fully_shard(block, ...)
    fully_shard(model, ...)
    trainer = Trainer(model, optimizer, config, loaders)
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from llm.config.training_config import TrainingConfig
from llm.training.scheduler import WarmupCosineSchedule, ConstantSchedule
from llm.training.graceful import GracefulStopper
from llm.training.logger import TrainingLogger


class Trainer:
    """极简训练编排器。

    不做的事（交给 PyTorch）:
    - 不包装 DDP/FSDP
    - 不管理 autocast
    - 不管理 GradScaler
    - 不管理 backward 通信时机
    - 不管理数据分片
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        train_loader,          # DataLoader — 已 pin_memory
        val_loader,            # DataLoader
        raw_model: nn.Module | None = None,  # DDP 包装前的原始模型（checkpoint 用）
        scaler: torch.amp.GradScaler | None = None,
        ddp: bool = False,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.raw_model = raw_model or model
        self.scaler = scaler
        self.ddp = ddp

        # 持久迭代器
        self._train_iter = iter(train_loader)
        self._val_iter = iter(val_loader)

        # LR 调度
        if config.decay_lr:
            self.scheduler = WarmupCosineSchedule(
                lr_max=config.learning_rate,
                lr_min=config.min_lr,
                warmup_iters=config.warmup_iters,
                total_iters=config.lr_decay_iters,
            )
        else:
            self.scheduler = ConstantSchedule(config.learning_rate)

        # 训练状态
        self.iter_num = 0
        self.best_val_loss = float('inf')
        self._last_loss = 0.0

        # 优雅退出
        self.stopper = GracefulStopper(config.out_dir)

        # 日志
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.logger = TrainingLogger(config, self, self.raw_model) if rank == 0 else None

    @property
    def backend_device(self):
        """推断 device（兼容 DDP wrapped model）。"""
        # DDP: model.module.xxx.parameters()
        # FSDP: model.xxx.parameters()
        return next(self.model.parameters()).device

    @property
    def world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def is_master(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    # ═══════════════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════════════

    def train(self):
        config = self.config
        device = self.backend_device
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        use_amp = device_type == 'cuda' and config.dtype != 'float32'
        amp_dtype = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }.get(config.dtype, torch.float16)

        if self.logger:
            self.logger.start_gpu_monitor()

        X, Y = self._next_batch('train')
        t0 = time.time()

        while True:
            # ── 1. LR ──
            lr = self.scheduler(self.iter_num)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            # ── 2. 评估 + checkpoint ──
            if self.iter_num % config.eval_interval == 0 and self.is_master:
                self._evaluate_and_checkpoint()

            if self.iter_num == 0 and config.eval_only:
                break

            # ── 3. 梯度累积 ──
            for micro_step in range(config.gradient_accumulation_steps):
                is_last = (micro_step == config.gradient_accumulation_steps - 1)

                if self.stopper.should_stop():
                    break

                # 前向（autocast 根据 dtype 自动决定）
                with torch.amp.autocast(device_type, dtype=amp_dtype) if use_amp else torch.no_grad():  # noqa
                    # torch.no_grad() is wrong here, we need gradients
                    pass

                # 实际的前向
                ctx = (torch.amp.autocast(device_type, dtype=amp_dtype)
                       if use_amp else nullcontext())
                with ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / config.gradient_accumulation_steps

                loss_val = loss.detach()

                # 预取
                try:
                    X, Y = self._next_batch('train')
                except RuntimeError as e:
                    if "DataLoader worker" in str(e):
                        print("[WARNING] DataLoader worker 异常退出")
                        self.stopper._stop_requested = True
                        break
                    raise

                # 反向（DDP: no_sync 跳过 all-reduce）
                if self.scaler is not None:
                    scaled_loss = self.scaler.scale(loss)
                else:
                    scaled_loss = loss

                if self.ddp and not is_last:
                    with self.model.no_sync():
                        scaled_loss.backward()
                else:
                    scaled_loss.backward()

            self._last_loss = loss_val.item() if loss_val is not None else 0.0

            # 微批次不完整 → 跳过更新
            if micro_step < config.gradient_accumulation_steps - 1:
                continue

            # ── 4. 梯度裁剪 + 更新 ──
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), config.grad_clip
                )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            # ── 5. 日志 ──
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if self.iter_num % config.log_interval == 0 and self.is_master:
                self._log(dt)

            self.iter_num += 1

            if self.iter_num > config.max_iters or self.stopper.should_stop():
                if self.stopper._stop_requested and self.is_master:
                    self._save_checkpoint()
                    print(f"[INFO] 优雅退出，checkpoint 已保存 (iter {self.iter_num})")
                break

        if dist.is_initialized():
            dist.barrier()
        if self.logger:
            self.logger.stop_gpu_monitor()

    # ═══════════════════════════════════════════════════════════
    # 数据
    # ═══════════════════════════════════════════════════════════

    def _next_batch(self, split: str):
        it = self._train_iter if split == 'train' else self._val_iter
        X, Y = next(it)
        device = self.backend_device
        if device.type == 'cuda':
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)
        return X, Y

    # ═══════════════════════════════════════════════════════════
    # 评估 / checkpoint / 日志  (与 v1 相同，省略)
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def _evaluate_and_checkpoint(self):
        losses = self._estimate_loss()
        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}")

        if self.logger is not None:
            lr = self.scheduler(self.iter_num)
            self.logger.write_status_file(
                self.iter_num, losses['train'], lr, self.best_val_loss
            )

        if losses['val'] < self.best_val_loss or self.config.always_save_checkpoint:
            self.best_val_loss = losses['val']
            if self.iter_num > 0:
                self._save_checkpoint()

    @torch.no_grad()
    def _estimate_loss(self) -> dict:
        self.model.eval()
        out = {}
        device_type = 'cuda' if self.backend_device.type == 'cuda' else 'cpu'
        use_amp = device_type == 'cuda' and self.config.dtype != 'float32'
        amp_dtype = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }.get(self.config.dtype, torch.float16)

        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self._next_batch(split)
                ctx = (torch.amp.autocast(device_type, dtype=amp_dtype)
                       if use_amp else nullcontext())
                with ctx:
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def _save_checkpoint(self):
        config = self.config
        os.makedirs(config.out_dir, exist_ok=True)
        checkpoint = {
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.raw_model.config.to_dict(),
            'config': config.to_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
        print(f"saving checkpoint to {config.out_dir}/ckpt.pt")

    def load_checkpoint(self) -> int:
        ckpt_path = os.path.join(self.config.out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print(f"no checkpoint at {ckpt_path}")
            return 0
        ckpt = torch.load(ckpt_path, map_location=self.backend_device)
        state_dict = ckpt['model']
        for k in list(state_dict.keys()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
        self.raw_model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.iter_num = ckpt['iter_num']
        self.best_val_loss = ckpt['best_val_loss']
        print(f"resumed from {ckpt_path} at iter {self.iter_num}")
        return self.iter_num

    def _log(self, dt: float):
        lossf = self._last_loss * self.config.gradient_accumulation_steps
        lr = self.scheduler(self.iter_num)
        if self.iter_num >= 5 and self.logger is not None:
            import math
            mfu = self.raw_model.estimate_mfu(
                self.config.gradient_accumulation_steps
                * self.config.batch_size
                * self.raw_model.config.block_size
                * self.world_size,
                dt
            )
            mfu *= 312e12 / self.logger._mfu_baseline
        else:
            mfu = 0.0

        if self.logger is not None:
            self.logger.log_step(self.iter_num, lossf, dt, lr, mfu)
            if self.iter_num % 10 == 0:
                self.logger.write_status_file(
                    self.iter_num, lossf, lr, self.best_val_loss
                )


def nullcontext():
    """兼容 Python < 3.10 的 nullcontext。"""
    from contextlib import contextmanager
    @contextmanager
    def _null():
        yield
    return _null()
