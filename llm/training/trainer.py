"""Trainer —— 后端无关的训练循环编排器。

所有分布式/混合精度逻辑通过 TrainingBackend 抽象，
Trainer 只关心训练流程本身：数据 → 前向 → 反向 → 更新 → 评估 → 保存。

用法:
    from llm import GPT, GPTConfig, TrainingConfig, Trainer, create_backend

    config = TrainingConfig(max_iters=100, dataset="shakespeare_char")
    model_config = GPTConfig.from_preset("gpt2")
    model = GPT(model_config)
    backend = create_backend(config)
    trainer = Trainer(model, config, backend)
    trainer.train()
"""

import os
import time
import torch
import torch.nn as nn

from llm.config.model_config import GPTConfig
from llm.config.training_config import TrainingConfig
from llm.training.backends import TrainingBackend
from llm.training.scheduler import WarmupCosineSchedule, ConstantSchedule


class Trainer:
    """训练编排器。

    职责:
    - 编排训练循环（数据 → 前向 → 反向 → 更新）
    - LR 调度
    - 评估 + checkpoint 保存/恢复
    - 日志（文本 + 可选 wandb）
    - MFU 估算

    不负责（由 Backend 处理）:
    - 设备放置
    - 混合精度策略
    - 分布式通信
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        backend: TrainingBackend,
        data_provider=None,  # 见 llm/data/dataset.py
    ):
        self.config = config
        self.backend = backend
        self.data = data_provider

        # ── 模型准备（设备放置 + 分布式包装）──
        self.raw_model = model  # 保持原始引用（checkpoint 保存用）
        self.model = backend.prepare_model(model)

        # ── torch.compile ──
        if config.compile and hasattr(torch, 'compile'):
            try:
                import triton  # noqa: F401 — torch.compile 的 inductor 后端依赖 Triton
            except ImportError:
                print("[WARNING] Triton not available (e.g. on Windows), "
                      "skipping torch.compile. Install via: pip install triton")
            else:
                print("compiling the model... (takes ~a minute)")
                self.model = torch.compile(self.model)

        # ── 优化器 ──
        self.optimizer = backend.prepare_optimizer(
            self.model, config  # 传入 wrapped model
        )

        # ── LR 调度器 ──
        if config.decay_lr:
            self.scheduler = WarmupCosineSchedule(
                lr_max=config.learning_rate,
                lr_min=config.min_lr,
                warmup_iters=config.warmup_iters,
                total_iters=config.lr_decay_iters,
            )
        else:
            self.scheduler = ConstantSchedule(config.learning_rate)

        # ── 训练状态 ──
        self.iter_num = 0
        self.best_val_loss = float('inf')
        self.running_mfu = -1.0
        self._last_loss = 0.0

        # ── 日志 ──
        self.wandb = None
        if config.wandb_log and backend.is_master:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name or f"run_{int(time.time())}",
                    config=config.to_dict(),
                )
            except ImportError:
                print("[WARNING] wandb not installed, skipping logging")

    def train(self):
        """主训练循环。

        标准流程（后端无关）:
        1. 设置 LR
        2. 评估 + checkpoint（按 interval）
        3. 梯度累积循环:
           a. autocast 前向
           b. backward（后端控制同步时机）
           c. 异步预取下一 batch
        4. unscale → clip → step → zero_grad
        5. 日志
        """
        config = self.config
        backend = self.backend

        # 预取第一个 batch
        X, Y = self._get_batch('train')
        t0 = time.time()

        while True:
            # ── 1. 设置学习率 ──
            lr = self.scheduler(self.iter_num)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            # ── 2. 评估 + Checkpoint ──
            if self.iter_num % config.eval_interval == 0 and backend.is_master:
                self._evaluate_and_checkpoint()

            # eval_only 模式：评估一次就退出
            if self.iter_num == 0 and config.eval_only:
                break

            # ── 3. 梯度累积循环 ──
            for micro_step in range(config.gradient_accumulation_steps):
                is_last = (micro_step == config.gradient_accumulation_steps - 1)

                with backend.autocast_context():
                    logits, loss = self.model(X, Y)
                    loss = loss / config.gradient_accumulation_steps

                # 记录最后一个 micro_step 的 loss（用于日志打印）
                self._last_loss = loss.item()

                # 异步预取下一 batch（在 GPU 忙于 backward 时加载数据）
                X, Y = self._get_batch('train')

                # 反向传播（DDP: 非最后步通过 model.no_sync() 跳过 all-reduce）
                backend.backward(self.model, loss, is_last_micro_step=is_last)

            # ── 4. 梯度裁剪 → 更新参数 → 清零 ──
            if config.grad_clip > 0.0:
                backend.unscale(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), config.grad_clip
                )
            backend.step(self.optimizer)
            backend.zero_grad(self.optimizer)

            # ── 5. 日志 + MFU ──
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if self.iter_num % config.log_interval == 0 and backend.is_master:
                self._log(dt)

            self.iter_num += 1

            # ── 终止条件 ──
            if self.iter_num > config.max_iters:
                break

        backend.cleanup()

    # ═══════════════════════════════════════════════════════════════════
    # 评估 & Checkpoint
    # ═══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _evaluate_and_checkpoint(self):
        """评估 train/val loss，保存 checkpoint。"""
        config = self.config
        backend = self.backend

        losses = self._estimate_loss()
        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}")

        # Wandb 日志
        if self.wandb is not None:
            self.wandb.log({
                "iter": self.iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": self.scheduler(self.iter_num),
                "mfu": self.running_mfu * 100,
            })

        # 保存 checkpoint
        if losses['val'] < self.best_val_loss or config.always_save_checkpoint:
            self.best_val_loss = losses['val']
            if self.iter_num > 0:
                self._save_checkpoint()

    @torch.no_grad()
    def _estimate_loss(self) -> dict:
        """在 train/val 上采样评估 loss。"""
        config = self.config
        backend = self.backend

        self.model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                X, Y = self._get_batch(split)
                with backend.autocast_context():
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def _save_checkpoint(self):
        """保存完整训练状态到磁盘。"""
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
        ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
        torch.save(checkpoint, ckpt_path)
        print(f"saving checkpoint to {ckpt_path}")

    def load_checkpoint(self) -> int:
        """从 checkpoint 恢复训练状态。返回恢复时的 iter_num。"""
        config = self.config
        ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')

        if not os.path.exists(ckpt_path):
            print(f"no checkpoint found at {ckpt_path}, starting from scratch")
            return 0

        checkpoint = torch.load(ckpt_path, map_location=self.backend.device)

        # 恢复模型权重
        state_dict = checkpoint['model']
        # 兼容: 移除 torch.compile 产生的 _orig_mod. 前缀
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.raw_model.load_state_dict(state_dict)

        # 恢复优化器
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # 恢复训练状态
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"resumed from {ckpt_path} at iter {self.iter_num}")
        return self.iter_num

    # ═══════════════════════════════════════════════════════════════════
    # 日志 & MFU
    # ═══════════════════════════════════════════════════════════════════

    def _log(self, dt: float):
        """打印一步的训练指标。"""
        config = self.config

        # 计算 MFU（跳过前几步让训练稳定）
        if self.iter_num >= 5:
            fwdbwd_per_iter = (
                config.gradient_accumulation_steps
                * config.batch_size
                * self.raw_model.config.block_size
                * self.backend.world_size
            )
            mfu = self.raw_model.estimate_mfu(fwdbwd_per_iter, dt)
            self.running_mfu = (
                mfu
                if self.running_mfu == -1.0
                else 0.9 * self.running_mfu + 0.1 * mfu
            )

        lossf = self._last_loss * config.gradient_accumulation_steps
        print(
            f"iter {self.iter_num}: loss {lossf:.4f}, "
            f"time {dt*1000:.2f}ms, mfu {self.running_mfu*100:.2f}%"
        )

    # ═══════════════════════════════════════════════════════════════════
    # 数据获取
    # ═══════════════════════════════════════════════════════════════════

    def _get_batch(self, split: str):
        """从 DataProvider 获取一个 batch，并移动到正确设备。

        如果还没有 data_provider，则回退到内存映射方式（兼容 nanoGPT 的 .bin 文件）。
        """
        if self.data is not None:
            X, Y = self.data.get_batch(split)

            # 移动到设备 + 锁页内存（异步传输）
            device_type = 'cuda' if self.backend.device.type == 'cuda' else 'cpu'
            if device_type == 'cuda':
                X = X.pin_memory().to(self.backend.device, non_blocking=True)
                Y = Y.pin_memory().to(self.backend.device, non_blocking=True)
            else:
                X, Y = X.to(self.backend.device), Y.to(self.backend.device)
            return X, Y

        # 回退: 如果没有注入 data_provider，需要子类覆盖此方法
        raise RuntimeError(
            "No data_provider set. Either pass one to Trainer.__init__ "
            "or override _get_batch()."
        )
