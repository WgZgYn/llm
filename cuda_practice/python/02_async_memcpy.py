# 02_async_memcpy.py —— 同步拷贝 vs 异步拷贝，以及 pageable vs pinned 内存
#
# 学习目标（对应 cuda_native/02_async_memcpy.cu）：
#   1. 普通 .to('cuda') / .copy_ 是"同步"的：主机阻塞到拷贝完成才返回。
#   2. non_blocking=True 是"异步"拷贝（等价 cudaMemcpyAsync）。
#   3. 但异步拷贝要真正异步，主机内存必须是 pinned（page-locked）。
#      pageable 内存下 non_blocking 会退化为同步 + 中转（staging），既慢又失去异步性。
#
# torch.cuda 映射：
#   cudaMemcpyAsync        -> tensor.to('cuda', non_blocking=True) / copy_(..., non_blocking=True)
#   cudaMallocHost(pinned) -> tensor.pin_memory() / torch.empty(..., pin_memory=True)
import time
import torch


def copy_h2d_bench(name, src, dst, non_blocking, warmup=3, iters=10):
    """测 H2D 拷贝带宽（GB/s）。src 是 CPU tensor，dst 是 GPU tensor。"""
    for _ in range(warmup):
        dst.copy_(src, non_blocking=non_blocking)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src, non_blocking=non_blocking)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    nbytes = src.numel() * src.element_size()
    gbs = nbytes / (ms * 1e6)   # bytes / ms / 1e6 -> GB/s
    print(f"  {name:<32}: {ms:8.3f} ms   {gbs:7.2f} GB/s")
    return ms, gbs


def main():
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    nbytes = 256 * 1024 * 1024        # 256 MB
    n = nbytes // 4                   # float32 元素数

    dst = torch.empty(n, dtype=torch.float32, device="cuda")

    # pageable（普通 CPU 内存）与 pinned（page-locked）
    src_pageable = torch.empty(n, dtype=torch.float32)          # pageable
    src_pinned = torch.empty(n, dtype=torch.float32, pin_memory=True)  # pinned

    print("===== H2D 拷贝 256 MB：pageable/pinned × 同步/异步 =====\n")

    copy_h2d_bench("同步 copy_  (pageable)", src_pageable, dst, non_blocking=False)
    copy_h2d_bench("异步 copy_  (pageable)", src_pageable, dst, non_blocking=True)
    copy_h2d_bench("异步 copy_  (pinned)  ", src_pinned, dst, non_blocking=True)

    # 直接观察：non_blocking 的 pageable 拷贝是否真的不阻塞主机
    print("\n===== non_blocking 是否真的不阻塞主机？ =====")
    for name, src in [("pageable", src_pageable), ("pinned", src_pinned)]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)   # 理想情况下立刻返回
        t_return = (time.perf_counter() - t0) * 1e3
        print(f"  {name:<8} non_blocking 拷贝：主机立刻返回，用时 {t_return:.4f} ms "
              f"{'(小=真异步)' if t_return < 1 else '(大=其实被阻塞/同步化)'}")

    print("\n  结论要点：")
    print("  - pinned 内存让异步拷贝真正异步，带宽也更高（免去 staging 中转）。")
    print("  - pageable 内存下 non_blocking 退化为同步，异步收益消失。")
    print("  - 做异步流水线/重叠时，主机侧缓冲必须 pin_memory()。")

    del src_pageable, src_pinned


if __name__ == "__main__":
    main()
