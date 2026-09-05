# 05_sync_granularity.py —— 同步的粒度与代价  (torch.cuda 版)
#
# 学习目标（对应 cuda_native/05_sync_granularity.cu）：
#   三档同步粒度，越细代价越低、对并发破坏越小：
#     torch.cuda.synchronize()   等整个设备所有流  —— 最粗、最贵  (cudaDeviceSynchronize)
#     stream.synchronize()       等某一个流        —— 中等        (cudaStreamSynchronize)
#     event.synchronize()        等某一个"点"      —— 最细、最便宜 (cudaEventSynchronize)
#   另有"非阻塞查询"（query）与"阻塞等待"（synchronize）之分。
import time
import torch


def main():
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    torch.manual_seed(0)
    m = 2048
    a = torch.randn(m, m, device="cuda")
    b = torch.randn(m, m, device="cuda")
    x_comp = torch.rand(32 * 1024 * 1024, device="cuda")   # 128 MB，短计算用
    for _ in range(3):
        _ = torch.matmul(a, b)
        _ = torch.sin(x_comp)
    torch.cuda.synchronize()

    # ---- 1) 空闲同步的开销：设备空闲时，同步调用本身要花多少 -------------------
    N = 1000
    print(f"===== 1) 空闲同步调用开销（{N} 次平均）=====")
    s = torch.cuda.Stream()
    e = torch.cuda.Event()
    e.record()
    e.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        torch.cuda.synchronize()
    t_dev = (time.perf_counter() - t0) / N * 1e6

    t0 = time.perf_counter()
    for _ in range(N):
        s.synchronize()
    t_str = (time.perf_counter() - t0) / N * 1e6

    t0 = time.perf_counter()
    for _ in range(N):
        e.synchronize()
    t_evt = (time.perf_counter() - t0) / N * 1e6

    print(f"  torch.cuda.synchronize() : {t_dev:7.3f} us/次  (等整个设备)")
    print(f"  stream.synchronize()     : {t_str:7.3f} us/次  (等一个流)")
    print(f"  event.synchronize()      : {t_evt:7.3f} us/次  (等一个点)")

    # ---- 2) 粒度语义：device 等所有流，stream 只等一个流 ------------------------
    print("\n===== 2) 同步粒度：device(等所有) vs stream(等一个) =====")
    # 长任务用"拷贝"（copy engine），短任务用"计算"（SM），避免资源争用干扰
    src = torch.empty(64 * 1024 * 1024, dtype=torch.float32, pin_memory=True)  # 256 MB
    dst = torch.empty_like(src, device="cuda")
    s_copy = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()
    with torch.cuda.stream(s_comp):           # 预热 s_comp（首次在新流上启动内核有额外开销）
        torch.sin(x_comp)
    s_comp.synchronize()

    with torch.cuda.stream(s_copy):
        dst.copy_(src, non_blocking=True)     # 长拷贝 ~20 ms（copy engine）
    with torch.cuda.stream(s_comp):
        torch.sin(x_comp)                     # 短计算 ~1 ms（SM，元素级无 autotune）

    t0 = time.perf_counter()
    s_comp.synchronize()          # 只等短计算（~2 ms）
    t_sync_short = (time.perf_counter() - t0) * 1e3
    print(f"  s_comp.synchronize()     用时 {t_sync_short:6.3f} ms（只等计算，快）")

    with torch.cuda.stream(s_copy):
        dst.copy_(src, non_blocking=True)     # 再放一个长拷贝
    t0 = time.perf_counter()
    torch.cuda.synchronize()       # 等所有流（含长拷贝 ~20 ms）
    t_dev_sync = (time.perf_counter() - t0) * 1e3
    print(f"  torch.cuda.synchronize() 用时 {t_dev_sync:6.3f} ms（等所有流，含长拷贝）")
    del src, dst

    # ---- 3) 阻塞同步 vs 轮询查询 ----------------------------------------------
    print("\n===== 3) 阻塞同步 vs 轮询（query）=====")
    sp = torch.cuda.Stream()
    with torch.cuda.stream(sp):
        torch.matmul(a, b)          # 预热 sp 上的 matmul（避免 autotune 噪声）
    sp.synchronize()

    with torch.cuda.stream(sp):
        torch.matmul(a, b)
    t0 = time.perf_counter()
    sp.synchronize()
    t_block = (time.perf_counter() - t0) * 1e3

    with torch.cuda.stream(sp):
        torch.matmul(a, b)
    polls = 0
    t0 = time.perf_counter()
    while not sp.query():          # 非阻塞轮询，主机可边等边做别的
        polls += 1
    t_poll = (time.perf_counter() - t0) * 1e3

    print(f"  阻塞 stream.synchronize() : {t_block:6.3f} ms（干等）")
    print(f"  轮询 stream.query()       : {t_poll:6.3f} ms（查了 {polls} 次，期间可干别的）")


if __name__ == "__main__":
    main()
