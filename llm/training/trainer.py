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
from llm.training.graceful import GracefulStopper
from llm.training.logger import TrainingLogger


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
        data_provider=None,  # DataProvider | dict[str, CyclingDataLoader]
    ):
        self.config = config
        self.backend = backend
        # 兼容新旧两种数据接口
        if isinstance(data_provider, dict):
            # 新接口: DataLoader → 创建持久迭代器
            self._train_iter = iter(data_provider['train'])
            self._val_iter = iter(data_provider['val'])
            self._loaders = data_provider
            self.data = None
        else:
            self._train_iter = None
            self._val_iter = None
            self._loaders = None
            self.data = data_provider                     # 旧 DataProvider 接口

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

        # ── 优雅退出 ──
        self.stopper = GracefulStopper(config.out_dir)

        # ── 日志 ──
        self.logger = TrainingLogger(config, backend, self.raw_model) if backend.is_master else None
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

        # 启动 GPU 监控
        if self.logger is not None:
            self.logger.start_gpu_monitor()

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
            micro_step = 0
            loss_val = None  # 防止循环体未执行时未定义
            for micro_step in range(config.gradient_accumulation_steps):
                is_last = (micro_step == config.gradient_accumulation_steps - 1)

                # 检查退出信号（Ctrl-C 可能在微批次循环中触发）
                if self.stopper.should_stop():
                    break

                with backend.autocast_context():
                    logits, loss = self.model(X, Y)
                    loss = loss / config.gradient_accumulation_steps

                # detach 保存 loss 值（不触发 GPU 同步，异步流水线不被打断）
                loss_val = loss.detach()

                # 异步预取下一 batch（与 GPU backward 并行）
                try:
                    X, Y = self._get_batch('train')
                except RuntimeError as e:
                    if "DataLoader worker" in str(e):
                        print(f"\n[WARNING] DataLoader worker 异常退出，可能是 Ctrl-C")
                        self.stopper._stop_requested = True
                        break
                    raise

                # 反向传播（DDP: 非最后步通过 model.no_sync() 跳过 all-reduce）
                backend.backward(self.model, loss, is_last_micro_step=is_last)

            # 整个梯度累积完成后才同步取 loss（只 sync 一次）
            self._last_loss = loss_val.item() if loss_val is not None else 0.0

            # 微批次循环提前退出 → 梯度不完整 → 跳过参数更新
            if micro_step < config.gradient_accumulation_steps - 1:
                if not self.stopper._stop_requested:
                    self._last_loss = 0.0
                continue  # 回到 while True 开头，触发终止条件退出

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
            if self.iter_num > config.max_iters or self.stopper.should_stop():
                if self.stopper._stop_requested:
                    if backend.is_master:
                        self._save_checkpoint()
                    # 非 master 等一等 master 保存完
                    if backend.world_size > 1:
                        import torch.distributed as dist
                        dist.barrier()
                    if backend.is_master:
                        print(f"[INFO] 优雅退出，checkpoint 已保存 (iter {self.iter_num})")
                break

        backend.cleanup()
        self.stopper.cleanup()
        if self.logger is not None:
            self.logger.stop_gpu_monitor()

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

        # 写入状态文件
        if self.logger is not None:
            lr = self.scheduler(self.iter_num)
            self.logger.write_status_file(
                self.iter_num, losses['train'], lr, self.best_val_loss
            )

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
            out[split] = losses.mean().item()
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
        """打印一步的训练指标（委托给 TrainingLogger）。"""
        lossf = self._last_loss * self.config.gradient_accumulation_steps
        lr = self.scheduler(self.iter_num)

        # 计算 MFU
        if self.iter_num >= 5 and self.logger is not None:
            mfu = self.logger.compute_mfu(dt)
            self.running_mfu = mfu
        else:
            self.running_mfu = -1.0

        if self.logger is not None:
            self.logger.log_step(self.iter_num, lossf, dt, lr,
                                 max(self.running_mfu, 0.0))
            # 每 10 步更新一次状态文件
            if self.iter_num % 10 == 0:
                self.logger.write_status_file(
                    self.iter_num, lossf, lr, self.best_val_loss
                )

    # ═══════════════════════════════════════════════════════════════════
    # 数据获取
    # ═══════════════════════════════════════════════════════════════════

    def _get_batch(self, split: str):
        """获取一个 batch 并异步传输到 GPU。

        数据源:
        - DataLoader + IterableDataset: 已 pin_memory，直接 .to(device, non_blocking)
        - DataProvider (旧接口): 手动 pin_memory + 传输
        """
        device = self.backend.device
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'

        if self._train_iter is not None:
            # 新接口: DataLoader 持久迭代器（pin_memory 已由 DataLoader 处理）
            it = self._train_iter if split == 'train' else self._val_iter
            X, Y = next(it)
        elif self.data is not None:
            # 旧接口: DataProvider
            X, Y = self.data.get_batch(split)
        else:
            raise RuntimeError(
                "No data source set. Pass data_provider or loaders dict to Trainer."
            )

        if device_type == 'cuda':
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)
        return X, Y