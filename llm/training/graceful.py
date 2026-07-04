"""优雅退出机制。

支持三种触发方式:
    1. 在 out_dir 创建 STOP 文件
    2. SIGTERM 信号（如 kill <pid>）
    3. SIGINT 信号（Ctrl-C），连续两次则立即退出

用法:
    from llm.training.graceful import GracefulStopper

    stopper = GracefulStopper(out_dir)
    while training:
        ...
        if stopper.should_stop():
            save_checkpoint()
            break
"""

import os
import signal
import time
from pathlib import Path


class GracefulStopper:
    """优雅停止 —— 信号文件 + 信号处理器。

    触发后会在当前 step 完成后保存 checkpoint 再退出，
    而非暴力中断导致 checkpoint 损坏。

    连续两次 SIGINT（Ctrl-C）会立即退出，不保存（用于紧急情况）。
    """

    def __init__(self, out_dir: str):
        self.stop_file = Path(out_dir) / "STOP"
        self._stop_requested = False
        self._sigint_count = 0

        # 启动时清理旧的 STOP 文件
        if self.stop_file.exists():
            self.stop_file.unlink()
            print(f"[INFO] 已清理旧的 STOP 文件: {self.stop_file}")

        # 注册信号处理器
        self._original_sigint = signal.signal(signal.SIGINT, self._on_signal)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._on_signal)

    # ── 信号处理 ──

    def _on_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name

        if signum == signal.SIGINT:
            self._sigint_count += 1
            if self._sigint_count >= 2:
                print(f"\n[WARNING] 连续两次 {sig_name}，立即退出（不保存 checkpoint）")
                # 恢复原始信号处理器后硬退出
                signal.signal(signal.SIGINT, self._original_sigint)
                os._exit(1)

        print(f"\n[INFO] 收到 {sig_name} 信号，完成当前 step 后保存 checkpoint 退出...")
        print(f"        再按一次 Ctrl-C 强制退出（跳过保存）")
        self._stop_requested = True

    # ── 轮询 ──

    def should_stop(self) -> bool:
        """检查是否应该停止。在训练循环中周期性调用。"""
        if self._stop_requested:
            return True
        if self.stop_file.exists():
            print(f"\n[INFO] 检测到 {self.stop_file}，完成当前 step 后保存 checkpoint 退出...")
            self.stop_file.unlink()
            self._stop_requested = True
        return self._stop_requested

    # ── 清理 ──

    def cleanup(self):
        """恢复原始信号处理器。"""
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        if self.stop_file.exists():
            self.stop_file.unlink()
