"""增强训练日志：时间戳、GPU 信息、显存、ETA、token/s、MFU。

设计:
    - TrainingLogger: 核心日志类，在 Trainer 中集成
    - GPU_DATABASE: 主流 GPU 的 bf16/fp16 峰值 TFLOPS，用于 MFU 基准
    - 支持 pynvml（可选）获取 GPU 利用率
    - 每步日志写入 train.log，状态写入 status.json

用法:
    from llm.training.logger import TrainingLogger

    logger = TrainingLogger(config, backend, raw_model)
    logger.log_step(iter_num, loss, dt, lr, mfu)
    logger.write_status_file(iter_num, loss, lr, best_val)
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# GPU 数据库 —— bf16/fp16 峰值 TFLOPS + 显存
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPUInfo:
    name: str
    bf16_tflops: float    # bfloat16 峰值 TFLOPS（0 = 不支持）
    fp16_tflops: float    # float16 峰值 TFLOPS
    memory_gib: float     # 总显存 GiB


GPU_DATABASE = {
    # NVIDIA Data Center
    "H100":      GPUInfo("H100-SXM-80GB",   990, 990, 80),
    "H800":      GPUInfo("H800-SXM-80GB",   990, 990, 80),
    "A100":      GPUInfo("A100-SXM-80GB",   312, 312, 80),
    "A100-SXM":  GPUInfo("A100-SXM-40GB",   312, 312, 40),
    "A40":       GPUInfo("A40",             140, 140, 48),
    "A10":       GPUInfo("A10",              70,  70, 24),
    "A10G":      GPUInfo("A10G",             70,  70, 24),
    "L40S":      GPUInfo("L40S",            366, 366, 48),
    "L40":       GPUInfo("L40",             181, 181, 48),
    "L4":        GPUInfo("L4",              121, 121, 24),
    "V100":      GPUInfo("V100-SXM2-32GB",    0, 125, 32),
    "T4":        GPUInfo("T4",                0,  65, 16),

    # NVIDIA Consumer
    "RTX 5090":  GPUInfo("RTX 5090",        210, 210, 32),
    "RTX 4090":  GPUInfo("RTX 4090",        165, 165, 24),
    "RTX 4080":  GPUInfo("RTX 4080-SUPER",  130, 130, 16),
    "RTX 4070":  GPUInfo("RTX 4070",         75,  75, 12),
    "RTX 4060":  GPUInfo("RTX 4060",         45,  45,  8),
    "RTX 3090":  GPUInfo("RTX 3090",          0, 142, 24),
    "RTX 3080":  GPUInfo("RTX 3080",          0, 119, 10),
    "RTX 2080":  GPUInfo("RTX 2080-Ti",       0, 107, 11),

    # Apple Silicon (unified memory, rough estimates)
    "M2-Ultra":  GPUInfo("M2-Ultra",         27,  27, 192),
    "M3-Max":    GPUInfo("M3-Max",           20,  20, 128),
}

_UNKNOWN_DEFAULT = GPUInfo("Unknown-GPU", 80, 80, 16)


def _detect_gpu_by_name(gpu_name: str, device_type: str) -> GPUInfo:
    """通过 GPU 名称匹配数据库。"""
    if device_type != 'cuda' or not gpu_name:
        return GPUInfo("CPU", 0, 0, 0)

    for key, info in GPU_DATABASE.items():
        if key.lower() in gpu_name.lower():
            return info

    print(f"[WARNING] 未知 GPU: '{gpu_name}'，MFU 基准使用保守估算 (80 TFLOPS)。"
          f" 请在 GPU_DATABASE 中添加此型号。")
    return GPUInfo(gpu_name, 80, 80, 16)


# ═══════════════════════════════════════════════════════════════════════════════
# TrainingLogger
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """增强训练日志。

    集成到 Trainer 中:
        self.logger = TrainingLogger(config, backend, raw_model)
        self.logger.log_step(iter_num, loss, dt, lr, mfu)
        self.logger.write_status_file(iter_num, loss, lr, best_val)
    """

    def __init__(self, config, device_type: str, gpu_name: str,
                 raw_model, world_size: int, out_dir: str):
        import torch

        self.config = config
        self.model = raw_model
        self._world_size = world_size

        # ── GPU 检测 + MFU 基准 ──
        self.gpu_info = _detect_gpu_by_name(gpu_name, device_type)
        dtype = config.dtype
        if dtype == "bfloat16" and self.gpu_info.bf16_tflops == 0:
            print(f"[WARNING] {self.gpu_info.name} 不支持 bfloat16！"
                  f" 实际使用 fp32 路径，显存和速度都会受影响。"
                  f" 建议改用 dtype='float16'。")
        self._mfu_baseline = (
            self.gpu_info.bf16_tflops if dtype == "bfloat16"
            else self.gpu_info.fp16_tflops
        ) * 1e12  # → FLOPS

        # ── 输出路径 ──
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_path = os.path.join(self.out_dir, "train.log")
        self.status_path = os.path.join(self.out_dir, "status.json")

        # ── ETA 追踪 ──
        self.start_time = time.time()
        self._step_times: list[float] = []     # 最近 100 步耗时
        self._running_mfu: float = -1.0

        # ── GPU 利用率后台监控 ──
        self._gpu_util: float = 0.0
        self._device_idx: int = 0
        if device_type == 'cuda':
            self._device_idx = torch.cuda.current_device()
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # 写入日志头
        self._write_header()

    # ═══════════════════════════════════════════════════════════════════
    # 日志头
    # ═══════════════════════════════════════════════════════════════════

    def _write_header(self):
        """启动时写入一次环境信息。"""
        import torch

        bs = self.config.batch_size
        ga = self.config.gradient_accumulation_steps
        lines = [
            f"{'=' * 70}",
            f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"GPU: {self.gpu_info.name} ({self.gpu_info.memory_gib:.0f} GiB)"
            f" × {self._world_size}",
            f"dtype: {self.config.dtype} | backend: {self.config.backend}",
            f"model: {self.model.config.n_layer} layers, "
            f"{self.model.config.n_head} heads, "
            f"{self.model.config.n_embd} dim",
            f"params: {self.model.get_num_params()/1e6:.2f}M",
            f"total batch: {bs} × {ga} × {self._world_size} = "
            f"{bs * ga * self._world_size}",
            f"MFU baseline: {self._mfu_baseline/1e12:.0f} TFLOPS "
            f"({'bf16' if self.config.dtype == 'bfloat16' else 'fp16'})",
            f"{'=' * 70}",
            "",
        ]

        # 终端输出
        for line in lines:
            if line:
                print(line)

        # 写入日志文件
        with open(self.log_path, "w") as f:
            for line in lines:
                if line:
                    f.write(line + "\n")

    # ═══════════════════════════════════════════════════════════════════
    # 每步日志
    # ═══════════════════════════════════════════════════════════════════

    def log_step(self, iter_num: int, loss: float, dt: float,
                 lr: float, mfu: float):
        """每步日志行。"""
        config = self.config

        # ── 追踪 ETA ──
        self._step_times.append(dt)
        if len(self._step_times) > 100:
            self._step_times.pop(0)

        # ── 显存 ──
        import torch
        mem_alloc = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        mem_rsvd_extra = mem_reserved - mem_alloc

        # ── Tokens/sec ──
        tokens_per_step = (
            config.gradient_accumulation_steps
            * config.batch_size
            * self.model.config.block_size
            * self._world_size
        )
        tokens_per_sec = tokens_per_step / dt

        # ── ETA ──
        eta_str = self._compute_eta(iter_num)

        # ── 时间戳 ──
        now = datetime.now().strftime("%m-%d %H:%M:%S")

        # ── MFU 校正标记 ──
        mfu_note = ""
        if self.gpu_info.bf16_tflops == 0 and config.dtype == "bfloat16":
            mfu_note = " [WARN: bf16 on non-bf16 GPU]"

        # ── 进度 ──
        pct = 100.0 * iter_num / config.max_iters if config.max_iters > 0 else 0

        # ── 组装 ──
        line = (
            f"[{now}] "
            f"iter {iter_num:>6d}/{config.max_iters} ({pct:5.1f}%) | "
            f"loss {loss:.4f} | "
            f"lr {lr:.2e} | "
            f"dt {dt*1000:.0f}ms | "
            f"tok/s {tokens_per_sec:,.0f} | "
            f"mfu {mfu*100:5.1f}%{mfu_note} | "
            f"vram {mem_alloc:.1f}/{self.gpu_info.memory_gib:.0f}GiB"
        )
        if mem_rsvd_extra > 0.5:
            line += f" (+{mem_rsvd_extra:.1f}GiB rsvd)"
        line += f" | gpu {self._gpu_util:.0f}%"
        line += f" | ETA {eta_str}"

        print(line)

        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _compute_eta(self, iter_num: int) -> str:
        """估算剩余时间。"""
        remaining = self.config.max_iters - iter_num
        if remaining <= 0 or len(self._step_times) < 5:
            return "--:--:--"

        avg_dt = sum(self._step_times) / len(self._step_times)
        total_seconds = int(remaining * avg_dt)

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        if days > 0:
            return f"{days}d {hours:02d}:{minutes:02d}h"
        elif hours > 0:
            return f"{hours}:{minutes:02d}:{total_seconds%60:02d}"
        else:
            return f"{minutes}:{total_seconds%60:02d}"

    # ═══════════════════════════════════════════════════════════════════
    # GPU 利用率后台监控（可选，需要 pynvml）
    # ═══════════════════════════════════════════════════════════════════

    def start_gpu_monitor(self, interval: float = 2.0):
        """启动后台线程，周期性采样 GPU 利用率。"""
        try:
            import pynvml
            pynvml.nvmlInit()
        except ImportError:
            return  # 静默跳过

        device_idx = self._device_idx

        def _loop():
            while not self._monitor_stop.is_set():
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self._gpu_util = util.gpu
                except Exception:
                    pass
                self._monitor_stop.wait(interval)

        self._monitor_thread = threading.Thread(target=_loop, daemon=True)
        self._monitor_thread.start()

    def stop_gpu_monitor(self):
        self._monitor_stop.set()

    # ═══════════════════════════════════════════════════════════════════
    # 状态文件（后台运行时外部监控用）
    # ═══════════════════════════════════════════════════════════════════

    def write_status_file(self, iter_num: int, loss: float, lr: float,
                          best_val: float):
        """写入 status.json（原子写），后台运行时 `cat` 或脚本读取。"""
        import torch

        elapsed = time.time() - self.start_time
        avg_dt = (
            sum(self._step_times) / len(self._step_times)
            if self._step_times else 0
        )

        status = {
            "timestamp": datetime.now().isoformat(),
            "host": os.uname().nodename if hasattr(os, "uname") else "",
            "iter": iter_num,
            "max_iters": self.config.max_iters,
            "progress_pct": round(100 * iter_num / max(1, self.config.max_iters), 2),
            "loss": round(loss, 4),
            "best_val_loss": round(best_val, 4),
            "lr": lr,
            "elapsed": str(timedelta(seconds=int(elapsed))),
            "eta": self._compute_eta(iter_num),
            "avg_dt_ms": round(avg_dt * 1000, 1),
            "mfu_pct": round(self._running_mfu * 100, 1),
            "gpu_name": self.gpu_info.name,
            "vram_alloc_gib": round(
                torch.cuda.memory_allocated() / 1024**3, 1
            ),
            "vram_total_gib": self.gpu_info.memory_gib,
            "gpu_util_pct": round(self._gpu_util, 1),
            "dtype": self.config.dtype,
            "backend": self.config.backend,
            "world_size": self._world_size,
        }

        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.status_path)   # 原子重命名

    # ═══════════════════════════════════════════════════════════════════
    # MFU
    # ═══════════════════════════════════════════════════════════════════

    def compute_mfu(self, dt: float) -> float:
        """使用检测到的 GPU 基准计算 MFU。"""
        config = self.config

        fwdbwd_per_iter = (
            config.gradient_accumulation_steps
            * config.batch_size
            * self.model.config.block_size
            * self._world_size
        )
        mfu = self.model.estimate_mfu(fwdbwd_per_iter, dt)

        # 修正: estimate_mfu() 硬编码 A100 312 TFLOPS → 用实际 GPU 基准
        mfu *= 312e12 / max(self._mfu_baseline, 1e12)

        # 指数平滑
        self._running_mfu = (
            mfu if self._running_mfu == -1.0
            else 0.9 * self._running_mfu + 0.1 * mfu
        )
        return self._running_mfu

    @property
    def running_mfu(self) -> float:
        return self._running_mfu
