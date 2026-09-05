# 06_stream_semantics.py —— 跨流依赖 & 默认流语义  (torch.cuda 版)
#
# 学习目标（对应 cuda_native/06_stream_semantics.cu）：
#   1. stream.wait_event(event)：让一个流等待另一个流里的"某个点"，
#      建立"精确"跨流依赖（比同步整个设备细得多）。
#      下面用"不等 vs 等"两组计时，量化这个依赖的效果。
#   2. torch 里所有 op 都跑在"当前流"上（默认是遗留默认流 stream 0）。
#      torch.cuda.Stream() 默认是非阻塞流，各流互不隐式同步；
#      需要先后关系时必须用 wait_event / wait_stream 显式声明。
#
# torch.cuda 映射：
#   cudaStreamWaitEvent -> stream.wait_event(event)
#   cudaEventRecord     -> event.record(stream)
import time
import torch


def main():
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    torch.manual_seed(0)

    # 短任务：元素级计算（SM）；长任务：拷贝（copy engine）——不同硬件，互不争抢。
    x_comp = torch.rand(32 * 1024 * 1024, device="cuda")      # 128 MB，短计算 ~1 ms
    src = torch.empty(64 * 1024 * 1024, dtype=torch.float32, pin_memory=True)  # 256 MB
    dst = torch.empty_like(src, device="cuda")

    for _ in range(3):
        torch.sin(x_comp)
    torch.cuda.synchronize()

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    with torch.cuda.stream(s2):
        torch.sin(x_comp)               # 预热 s2（避免首次切流开销）
    torch.cuda.synchronize()

    print("===== stream.wait_event：让 s2 等 s1 的某个\"点\" =====\n")

    # 情况 A：s2 不等 ev1 —— 立刻执行
    with torch.cuda.stream(s1):
        dst.copy_(src, non_blocking=True)   # s1 上长拷贝 ~20 ms
        evA = torch.cuda.Event()
        evA.record()                        # evA 落在 s1 上，拷贝之后

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(s2)
    with torch.cuda.stream(s2):
        torch.sin(x_comp)
    end.record(s2)
    torch.cuda.synchronize()
    t_no_wait = start.elapsed_time(end)
    print(f"  [不等] s2 上的短计算耗时 {t_no_wait:6.3f} ms（没等 s1 的长拷贝，立刻执行）")

    # 情况 B：s2 先等 evB —— 必须等 s1 的长拷贝完成
    # （重新放一个长拷贝：上一个拷贝已被上面的 synchronize 等完了）
    with torch.cuda.stream(s1):
        dst.copy_(src, non_blocking=True)   # 新的长拷贝 ~20 ms
        evB = torch.cuda.Event()
        evB.record()

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record(s2)             # s2 时间线起点（还没开始等）
    s2.wait_event(evB)            # 等价 cudaStreamWaitEvent(s2, evB, 0)
    with torch.cuda.stream(s2):
        torch.sin(x_comp)
    end2.record(s2)
    torch.cuda.synchronize()
    t_with_wait = start2.elapsed_time(end2)
    print(f"  [等]  s2 上的短计算耗时 {t_with_wait:6.3f} ms（先等 s1 的长拷贝完成）")
    print(f"  -> 差值 {t_with_wait - t_no_wait:6.3f} ms 就是 s2 等待 s1 的时间")

    print("\n  语义：wait_event 只让 s2 等 s1 的\"那个点\"(evB)，")
    print("        不等 s1 之后的新任务 —— 这就是\"精确到点\"的跨流依赖。")

    # ---- 当前流与默认流 -------------------------------------------------------
    print("\n===== 当前流 vs 默认流 =====")
    print(f"  torch.cuda.current_stream() : {torch.cuda.current_stream()}")
    print(f"  torch.cuda.default_stream() : {torch.cuda.default_stream()}")
    print("  torch 每个 op 都排在 current_stream 上；with torch.cuda.stream(s) 可临时切换。")
    print("  注：torch.cuda.Stream() 是非阻塞流，与默认流之间不隐式同步，")
    print("      因此事件必须显式 record(stream) 到对应流上，才能测到该流的时间线。")


if __name__ == "__main__":
    main()
