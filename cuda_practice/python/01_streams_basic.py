# 01_streams_basic.py —— 流（stream）的基本概念  (torch.cuda 版)
#
# 学习目标（与 cuda_native/01_streams_basic.cu 一一对应）：
#   1. 内核启动是"异步"的：主机把任务丢进队列后立刻返回，不等待执行完成。
#   2. 流（torch.cuda.Stream）是一个 FIFO 队列，同一流内的操作按提交顺序执行。
#   3. 不同流是相互独立的队列：一个流上的短任务不会等另一个流上的长任务。
#      （真正的"重叠"要看 03：拷贝走 copy engine、计算走 SM，两者才能并行。）
#
# torch.cuda 与 CUDA Runtime 的映射：
#   cudaStream_t / cudaStreamCreate      -> torch.cuda.Stream()
#   cudaStreamDestroy                   -> (GC 自动回收)
#   cudaEventRecord / cudaEventElapsedTime -> Event.record() / Event.elapsed_time()
#   cudaDeviceSynchronize               -> torch.cuda.synchronize()
import time
import torch


def main():
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    torch.manual_seed(0)
    n = 2048                       # matmul 尺寸（计算负载，可调）
    dev = "cuda"

    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)

    # 预热：第一次调用含驱动/JIT 初始化开销，先跑热
    x_comp = torch.rand(32 * 1024 * 1024, device=dev)   # 128 MB，短计算用
    for _ in range(3):
        _ = torch.matmul(a, b)
        _ = torch.sin(x_comp)
    torch.cuda.synchronize()

    # ---- 1) 内核启动是异步的 -------------------------------------------------
    print("===== 1) 内核启动是异步的 =====")
    t0 = time.perf_counter()
    c = torch.matmul(a, b)        # 只是把 kernel 排进当前流，立刻返回
    t1 = time.perf_counter()
    print(f"  matmul 启动（排队）仅用 {(t1 - t0) * 1e3:.4f} ms —— 主机没等它算完")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print(f"  matmul 实际执行耗时: {start.elapsed_time(end):.3f} ms（事件计时）")

    # ---- 2) 流是 FIFO 队列，流内按顺序执行 ------------------------------------
    print("\n===== 2) 流内操作按顺序（FIFO）执行 =====")
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s):
        torch.matmul(a, b)
        torch.matmul(a, b)
    torch.cuda.synchronize()
    two_serial = (time.perf_counter() - t0) * 1e3
    print(f"  同一流两个 matmul 总耗时: {two_serial:.3f} ms（≈ 2× 单次，说明串行）")

    # ---- 3) 不同流是相互独立的队列 --------------------------------------------
    print("\n===== 3) 不同流互不等待（独立队列）=====")
    # 关键：长任务用"拷贝"（走 copy engine / DMA），短任务用"计算"（走 SM）。
    # 若都用计算（matmul），长任务会占满 SM，短任务只能等它让出——那是"资源争用"
    # 而非"流依赖"。用不同硬件单元，才能真正观察到"两个流互不等待"。
    src = torch.empty(64 * 1024 * 1024, dtype=torch.float32, pin_memory=True)  # 256 MB
    dst = torch.empty_like(src, device=dev)
    s_copy = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()
    with torch.cuda.stream(s_comp):            # 预热 s_comp（首次在新流上启动内核有额外开销）
        torch.sin(x_comp)
    s_comp.synchronize()

    with torch.cuda.stream(s_copy):
        dst.copy_(src, non_blocking=True)      # 长拷贝 ~20 ms，走 copy engine
    with torch.cuda.stream(s_comp):
        torch.sin(x_comp)                      # 短计算 ~1 ms，走 SM（元素级，无 autotune 噪声）

    t0 = time.perf_counter()
    s_comp.synchronize()                       # 只等 s_comp（短计算）
    t_comp = (time.perf_counter() - t0) * 1e3
    print(f"  s_comp.synchronize() 用时 {t_comp:.3f} ms（只等短计算，快）")
    print(f"  此时拷贝还在跑吗？ s_copy.query() = {s_copy.query()}（False=仍在跑）")
    print("  -> 说明 s_comp 上的短计算没有等 s_copy 上的长拷贝：两个流相互独立")

    torch.cuda.synchronize()            # 清理
    del src, dst


if __name__ == "__main__":
    main()
