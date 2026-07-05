"""优雅退出机制。

支持三种触发方式:
    1. 在 out_dir 创建 STOP 文件
    2. SIGTERM 信号（如 kill <pid>）
    3. SIGINT 信号（Ctrl-C），连续两次则立即退出

DDP 安全: should_stop() 在所有 rank 上同步调用时，结果一致。

用法:
    from llm.training.graceful import GracefulStopper

    stopper = GracefulStopper(out_dir)
    while training:
        if stopper.is_stopping:
            save_checkpoint()
            break
        ...
        stopper.should_stop()   # 每步轮询
"""

import os
import signal
from pathlib import Path


class GracefulStopper:
    """优雅停止 —— 信号文件 + 信号处理器。

    公开 API:
        is_stopping:   是否已触发停止（读取此属性做清理）
        should_stop(): 轮询检查（信号文件 + 信号标志）
        cleanup():     恢复信号处理器
    """

    def __init__(self, out_dir: str):
        self.stop_file = Path(out_dir) / "STOP"
        self._stopping = False
        self._sigint_count = 0

        # 启动时清理旧的 STOP 文件
        if self.stop_file.exists():
            self.stop_file.unlink()
            print(f"[INFO] 已清理旧的 STOP 文件: {self.stop_file}")

        # 注册信号处理器
        self._orig_sigint = signal.signal(signal.SIGINT, self._on_signal)
        self._orig_sigterm = signal.signal(signal.SIGTERM, self._on_signal)

    # ── 公开属性 ──

    @property
    def is_stopping(self) -> bool:
        """是否已触发停止（供退出逻辑读取）。"""
        return self._stopping

    # ── 信号处理 ──

    def _on_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name

        if signum == signal.SIGINT:
            self._sigint_count += 1
            if self._sigint_count >= 2:
                print(f"\n[WARNING] 连续两次 {sig_name}，立即退出（不保存 checkpoint）")
                signal.signal(signal.SIGINT, self._orig_sigint)
                os._exit(1)

        print(f"\n[INFO] 收到 {sig_name} 信号，将在当前 step 完成后保存 checkpoint 退出...")
        print(f"        再按一次 Ctrl-C 强制退出（跳过保存）")
        self._stopping = True

    # ── 轮询 ──

    def should_stop(self) -> bool:
        """检查停止条件（信号文件 + 已收到的信号）。

        在训练循环中周期性调用，如微批次循环内。
        返回 True 后应跳出当前微批次循环，由外层 is_stopping 接管清理逻辑。
        """
        if self._stopping:
            return True
        if self.stop_file.exists():
            print(f"\n[INFO] 检测到 {self.stop_file}，"
                  f"将在当前 step 完成后保存 checkpoint 退出...")
            self.stop_file.unlink()
            self._stopping = True
        return self._stopping

    # ── 清理 ──

    def cleanup(self):
        """恢复原始信号处理器，清理 STOP 文件。"""
        signal.signal(signal.SIGINT, self._orig_sigint)
        signal.signal(signal.SIGTERM, self._orig_sigterm)
        if self.stop_file.exists():
            self.stop_file.unlink()
