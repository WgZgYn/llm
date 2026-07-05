"""Trainer —— 极简训练循环编排器。

不做的事（全部交给 PyTorch 和 train.py）:
    - 不包装 DDP/FSDP/DeepSpeed
    - 不创建 optimizer / scaler / scheduler
    - 不做数据分片

Trainer 只负责: 拿数据 → 前向 → 反向 → 更新 → 评估 → 日志 → checkpoint。
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
    """训练编排器。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        train_loader,
        val_loader,
        raw_model: nn.Module | None = None,
        scaler: torch.amp.GradScaler | None = None,
        ddp: bool = False,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.raw_model = raw_model or model
        self.scaler = scaler
        self.ddp = ddp

        self._train_iter = iter(train_loader)
        self._val_iter = iter(val_loader)

        if config.decay_lr:
            self.scheduler = WarmupCosineSchedule(
                lr_max=config.learning_rate, lr_min=config.min_lr,
                warmup_iters=config.warmup_iters,
                total_iters=config.lr_decay_iters,
            )
        else:
            self.scheduler = ConstantSchedule(config.learning_rate)

        self.iter_num = 0
        self.best_val_loss = float('inf')
        self._last_loss = 0.0

        self.stopper = GracefulStopper(config.out_dir)

        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.logger = (TrainingLogger(config, self._device_str, self._gpu_name,
                                      self.raw_model, self._world_size,
                                      config.out_dir)
                       if self._rank == 0 else None)

    # ── 属性 ──

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def _device_str(self) -> str:
        d = self.device
        return 'cuda' if d.type == 'cuda' else str(d)

    @property
    def _gpu_name(self) -> str:
        if self.device.type != 'cuda':
            return 'CPU'
        return torch.cuda.get_device_name(self.device)

    # ═══════════════════════════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════════════════════════

    def train(self):
        config = self.config
        device_type = self._device_str
        use_amp = device_type == 'cuda' and config.dtype != 'float32'
        amp_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(
            config.dtype, torch.float16)

        if self.logger:
            self.logger.start_gpu_monitor()

        X, Y = self._fetch('train')
        t0 = time.time()

        while True:
            # ── 1. LR ──
            lr = self.scheduler(self.iter_num)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            # ── 2. 评估 + checkpoint ──
            if self.iter_num % config.eval_interval == 0 and self._rank == 0:
                self._eval_and_save()

            if self.iter_num == 0 and config.eval_only:
                break

            # ── 3. 梯度累积 ──
            micro_step = 0
            loss_val = None
            for micro_step in range(config.gradient_accumulation_steps):
                if self.stopper.should_stop():
                    break

                # 前向
                with (torch.amp.autocast(device_type, dtype=amp_dtype)
                      if use_amp else _nullcontext()):
                    _, loss = self.model(X, Y)
                    loss = loss / config.gradient_accumulation_steps

                loss_val = loss.detach()

                # 预取
                try:
                    X, Y = self._fetch('train')
                except RuntimeError as e:
                    if "DataLoader worker" in str(e):
                        print(f"[rank{self._rank}] DataLoader worker 异常退出")
                        self.stopper._stopping = True
                        break
                    raise

                # 反向
                is_last = (micro_step == config.gradient_accumulation_steps - 1)
                s_loss = self.scaler.scale(loss) if self.scaler else loss
                if self.ddp and not is_last:
                    with self.model.no_sync():
                        s_loss.backward()
                else:
                    s_loss.backward()

            self._last_loss = loss_val.item() if loss_val is not None else 0.0

            # 微批次不完整 → 跳过参数更新，检查退出
            if micro_step < config.gradient_accumulation_steps - 1:
                if self.stopper.is_stopping:
                    self._shutdown()
                    return
                continue

            # ── 4. 梯度裁剪 + 更新 ──
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), config.grad_clip)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # ── 5. 日志 ──
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % config.log_interval == 0 and self._rank == 0:
                self._log(dt)

            self.iter_num += 1

            # ── 终止 ──
            if self.iter_num > config.max_iters or self.stopper.is_stopping:
                self._shutdown()
                return

    def _shutdown(self):
        """统一退出: DDP 同步 → checkpoint 保存 → 清理。"""
        if dist.is_initialized():
            dist.barrier()                    # 等所有 rank 到达此点
        if self._rank == 0 and self.iter_num > 0:
            self._save_checkpoint()
            print(f"[INFO] 已保存 checkpoint (iter {self.iter_num})")
        if dist.is_initialized():
            dist.barrier()                    # 等 rank-0 保存完成
        self.stopper.cleanup()
        if self.logger:
            self.logger.stop_gpu_monitor()

    # ═══════════════════════════════════════════════════════════════
    # 评估 & Checkpoint
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _eval_and_save(self):
        losses = self._estimate_loss()
        train_loss = losses['train'].item() if isinstance(losses['train'], torch.Tensor) else losses['train']
        val_loss = losses['val'].item() if isinstance(losses['val'], torch.Tensor) else losses['val']
        print(f"step {self.iter_num}: train loss {train_loss:.4f}, "
              f"val loss {val_loss:.4f}")

        lr = self.scheduler(self.iter_num)
        if self.logger:
            self.logger.write_status_file(
                self.iter_num, train_loss, lr, self.best_val_loss)

        if val_loss < self.best_val_loss or self.config.always_save_checkpoint:
            self.best_val_loss = val_loss
            if self.iter_num > 0:
                self._save_checkpoint()

    @torch.no_grad()
    def _estimate_loss(self) -> dict:
        self.model.eval()
        out = {}
        dtype = self.config.dtype
        use_amp = self._device_str == 'cuda' and dtype != 'float32'
        amp_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(
            dtype, torch.float16)

        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self._fetch(split)
                with (torch.amp.autocast(self._device_str, dtype=amp_dtype)
                      if use_amp else _nullcontext()):
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def _save_checkpoint(self):
        os.makedirs(self.config.out_dir, exist_ok=True)
        ckpt = {
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.raw_model.config.to_dict(),
            'config': self.config.to_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(ckpt, os.path.join(self.config.out_dir, 'ckpt.pt'))
        print(f"saving checkpoint to {self.config.out_dir}/ckpt.pt")

    def load_checkpoint(self) -> int:
        ckpt_path = os.path.join(self.config.out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print(f"no checkpoint at {ckpt_path}")
            return 0
        ckpt = torch.load(ckpt_path, map_location=self.device)
        sd = ckpt['model']
        for k in list(sd.keys()):
            if k.startswith('_orig_mod.'):
                sd[k[len('_orig_mod.'):]] = sd.pop(k)
        self.raw_model.load_state_dict(sd)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.iter_num = ckpt['iter_num']
        self.best_val_loss = ckpt['best_val_loss']
        print(f"resumed from {ckpt_path} at iter {self.iter_num}")
        return self.iter_num

    # ═══════════════════════════════════════════════════════════════
    # 数据 / 日志
    # ═══════════════════════════════════════════════════════════════

    def _fetch(self, split: str):
        it = self._train_iter if split == 'train' else self._val_iter
        X, Y = next(it)
        if self.device.type == 'cuda':
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)
        else:
            X, Y = X.to(self.device), Y.to(self.device)
        return X, Y

    def _log(self, dt: float):
        lossf = self._last_loss * self.config.gradient_accumulation_steps
        lr = self.scheduler(self.iter_num)
        if self.iter_num >= 5 and self.logger:
            mfu = self.logger.compute_mfu(dt)
        else:
            mfu = 0.0
        if self.logger:
            self.logger.log_step(self.iter_num, lossf, dt, lr, mfu)
            if self.iter_num % 10 == 0:
                self.logger.write_status_file(
                    self.iter_num, lossf, lr, self.best_val_loss)


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): pass
