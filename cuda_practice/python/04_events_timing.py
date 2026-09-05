# 04_events_timing.py —— 用事件（event）精确计时  (torch.cuda 版)
#
# 学习目标（对应 cuda_native/04_events_timing.cu）：
#   1. 事件（torch.cuda.Event）是"流上的一个标记/时间戳"。
#   2. 内核启动异步，主机时钟测不到真实执行时间，必须用事件。
#   3. 事件记录 GPU 时间线上的时刻，elapsed_time 读出两点间耗时。
#   4. 读耗时前必须等事件完成（synchronize）。
#   5. 不需要计时的同步用 Event(enable_timing=False) 更省。
#
# torch.cuda 映射：
#   cudaEventCreate / cudaEventRecord       -> torch.cuda.Event() / event.record()
#   cudaEventSynchronize / cudaEventQuery   -> event.synchronize() / event.query()
#   cudaEventElapsedTime                    -> start.elapsed_time(end)
import time
import torch


def main():
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    torch.manual_seed(0)
    m = 2048
    a = torch.randn(m, m, device="cuda")
    b = torch.randn(m, m, device="cuda")
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    print("===== 内核耗时：主机时钟 vs 事件计时 =====")

    # 错误示范：主机时钟测内核（异步 -> 只测到排队）
    t0 = time.perf_counter()
    c = torch.matmul(a, b)
    t1 = time.perf_counter()
    host_ms = (t1 - t0) * 1e3

    # 正确：两个事件 + elapsed_time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    event_ms = start.elapsed_time(end)

    print(f"  主机时钟(错误): {host_ms:.4f} ms   <- 只测到排队，不是执行")
    print(f"  事件计时(正确): {event_ms:.3f} ms   <- 真实执行时间")

    # ---- 读耗时前必须同步 -----------------------------------------------------
    print("\n===== 读 elapsed_time 前必须等事件完成 =====")
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e1.record()
    big = torch.randn(4096, 4096, device="cuda")   # 大 matmul，确保没算完
    torch.matmul(big, big)
    e2.record()
    print(f"  事件完成了吗？ e1.query()={e1.query()}  e2.query()={e2.query()}（False=还没完成）")
    torch.cuda.synchronize()
    print(f"  同步后 elapsed_time = {e1.elapsed_time(e2):.3f} ms")

    # ---- event.query()：非阻塞查询 vs synchronize()：阻塞 ----------------------
    print("\n===== query() 非阻塞 vs synchronize() 阻塞 =====")
    e3 = torch.cuda.Event()
    torch.matmul(a, b)
    e3.record()
    print(f"  记录后立刻 e3.query() = {e3.query()}（还没算完，False）")
    e3.synchronize()
    print(f"  synchronize 后 e3.query() = {e3.query()}（True）")

    # ---- enable_timing=False --------------------------------------------------
    print("\n===== Event(enable_timing=False) =====")
    e_sync = torch.cuda.Event(enable_timing=False)
    print("  禁用计时的 Event：仅作同步点，创建/记录开销更低；不能 elapsed_time。")


if __name__ == "__main__":
    main()
