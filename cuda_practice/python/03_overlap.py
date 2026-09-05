# 03_overlap.py —— 异步拷贝与内核计算的重叠（overlap）  (torch.cuda 版)
#
# 学习目标（对应 cuda_native/03_overlap.cu）：
#   拷贝走 GPU 的 copy engine（DMA），计算走 SM，是不同硬件单元，因此可重叠：
#   一边拷贝下一块数据，一边计算上一块数据。
#   前提：拷贝用 pinned 内存 + non_blocking=True，且拷贝/计算放在不同流上。
#   若放同一流（FIFO），会强制串行。
#
# torch.cuda 映射：
#   cudaStreamCreate / cudaMemcpyAsync -> torch.cuda.Stream + copy_(..., non_blocking=True)
#   with torch.cuda.stream(s)          -> 把后续 op 排到流 s 上
import time
import torch


def main():
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    torch.manual_seed(0)

    copy_bytes = 256 * 1024 * 1024      # 拷贝 256 MB
    n_elems = copy_bytes // 4
    m = 4096                            # matmul 尺寸（计算负载，约与 256MB 拷贝耗时相当）

    # 主机 pinned 缓冲（异步拷贝前提）
    src = torch.empty(n_elems, dtype=torch.float32, pin_memory=True)
    dst = torch.empty(n_elems, dtype=torch.float32, device="cuda")

    # 计算用的 GPU tensor（与拷贝的 dst 无关，避免数据依赖，重叠才"合法"）
    a = torch.randn(m, m, device="cuda")
    b = torch.randn(m, m, device="cuda")

    s_copy = torch.cuda.Stream()
    s_compute = torch.cuda.Stream()

    def copy_only():
        with torch.cuda.stream(s_copy):
            dst.copy_(src, non_blocking=True)

    def compute_only():
        with torch.cuda.stream(s_compute):
            torch.matmul(a, b)

    # 预热
    for _ in range(3):
        copy_only(); compute_only()
    torch.cuda.synchronize()

    # ---- 校准：单独测拷贝耗时、计算耗时 --------------------------------------
    torch.cuda.synchronize(); t0 = time.perf_counter(); copy_only(); torch.cuda.synchronize()
    t_copy = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize(); t0 = time.perf_counter(); compute_only(); torch.cuda.synchronize()
    t_compute = (time.perf_counter() - t0) * 1e3
    print(f"单独执行：拷贝 {t_copy:.3f} ms，计算 {t_compute:.3f} ms")
    print(f"若完全串行，总耗时 ≈ {t_copy + t_compute:.3f} ms\n")

    # ---- 情况 A：同一流 -> 串行 ------------------------------------------------
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.cuda.stream(s_copy):
        dst.copy_(src, non_blocking=True)
        torch.matmul(a, b)          # 同一流：拷贝结束后才开始算
    torch.cuda.synchronize()
    t_serial = (time.perf_counter() - t0) * 1e3

    # ---- 情况 B：拷贝/计算分两个流 -> 重叠 -------------------------------------
    torch.cuda.synchronize(); t0 = time.perf_counter()
    copy_only()     # s_copy 上异步拷贝
    compute_only()  # s_compute 上计算
    torch.cuda.synchronize()
    t_overlap = (time.perf_counter() - t0) * 1e3

    lower = max(t_copy, t_compute)
    print("===== 结果 =====")
    print(f"  同流串行     : {t_serial:8.3f} ms")
    print(f"  双流重叠     : {t_overlap:8.3f} ms")
    print(f"  理论重叠下界 : {lower:8.3f} ms  (= max(拷贝, 计算))")
    if t_serial > 0:
        print(f"  节省比例     : {100.0 * (t_serial - t_overlap) / t_serial:.1f}%")


if __name__ == "__main__":
    main()
