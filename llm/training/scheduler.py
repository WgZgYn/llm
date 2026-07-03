"""学习率调度器 —— Cosine 衰减 + Linear Warmup。

这是 LLM 预训练的标准 LR 调度策略（Chinchilla 论文推荐）。

用法:
    scheduler = WarmupCosineSchedule(lr_max=6e-4, lr_min=6e-5,
                                      warmup_iters=2000, total_iters=600000)
    lr = scheduler(iter_num)  # callable
"""

import math


class WarmupCosineSchedule:
    """Cosine 学习率衰减 + 线性 warmup。

    三个阶段:
    1. [0, warmup_iters):         线性从 0 上升到 lr_max
    2. [warmup_iters, total_iters): cosine 从 lr_max 衰减到 lr_min
    3. [total_iters, ∞):           保持在 lr_min

    用法:
        schedule = WarmupCosineSchedule(6e-4, 6e-5, 2000, 600000)
        for iter_num in range(max_iters):
            lr = schedule(iter_num)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
    """

    def __init__(
        self,
        lr_max: float = 6e-4,
        lr_min: float = 6e-5,
        warmup_iters: int = 2000,
        total_iters: int = 600000,
    ):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters

    def __call__(self, iter_num: int) -> float:
        """返回第 iter_num 步的学习率。"""
        # 阶段 1: Linear warmup
        if iter_num < self.warmup_iters:
            return self.lr_max * (iter_num + 1) / (self.warmup_iters + 1)

        # 阶段 3: 超过总步数 → 保持 min_lr
        if iter_num > self.total_iters:
            return self.lr_min

        # 阶段 2: Cosine 衰减
        decay_ratio = (iter_num - self.warmup_iters) / (
            self.total_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # [0, 1]
        return self.lr_min + coeff * (self.lr_max - self.lr_min)


class ConstantSchedule:
    """恒定学习率（用于微调或不使用衰减的场景）。"""

    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, iter_num: int) -> float:
        return self.lr
