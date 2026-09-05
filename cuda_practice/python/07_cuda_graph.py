# 07_cuda_graph.py —— CUDA Graph：消除大量小算子的"启动开销"
#
# 背景：
#   流内算子串行执行，但两个算子之间有一段"启动开销"（launch overhead）——
#   主机把 kernel 排进流、驱动再调度它上 GPU，单次约几 µs。当算子本身很小
#   （几 µs 就能算完）时，启动开销占比极高，GPU 大量时间在"空转等下一个 kernel"。
#
#   CUDA Graph 把一整段 kernel 序列"录制"成一个图，之后每次 replay 只花
#   1 次启动开销就把整段跑完，小算子场景能带来数倍加速。
#
# torch.cuda 映射：
#   cudaStreamBeginCapture/EndCapture -> with torch.cuda.graph(g):
#   cudaGraphInstantiate / cudaGraphLaunch -> g.replay()
import time
import torch


def main():
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    torch.manual_seed(0)
    dev = "cuda"

    N = 200                       # 小算子数量
    size = 64 * 1024              # 每个算子处理 64K 元素（很小，启动开销占比高）
    buf = torch.rand(size, device=dev)

    # 预热
    for _ in range(3):
        for _ in range(N):
            buf.sin_()
    torch.cuda.synchronize()

    # ---- 1) 普通：逐个启动 N 个小算子 -----------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        buf.sin_()                     # 每调用一次就是一次 kernel 启动
    torch.cuda.synchronize()
    t_plain = (time.perf_counter() - t0) * 1e3
    print(f"普通逐个启动 {N} 个 sin：{t_plain:.3f} ms（{N} 次启动开销 + 空转）")

    # ---- 2) CUDA Graph：录制一次，之后 replay 只花 1 次启动 -------------------
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):          # 进入"录制"模式，下面的 kernel 不执行、只录制
        for _ in range(N):
            buf.sin_()                 # 这 N 个 sin 被录进图里
    # 图录制完成。之后每次 g.replay() 一次性执行整张图。

    torch.cuda.synchronize()
    REPS = 50
    t0 = time.perf_counter()
    for _ in range(REPS):
        g.replay()
    torch.cuda.synchronize()
    t_graph = (time.perf_counter() - t0) * 1e3 / REPS

    print(f"CUDA Graph 单次 replay（含 {N} 个 kernel）：{t_graph:.3f} ms")
    if t_graph > 0:
        print(f"加速比：{t_plain / t_graph:.2f}x（省掉的就是启动开销）")

    # ---- 3) 注意事项 -----------------------------------------------------------
    print("\n注意：")
    print("  - 录制时 kernel 的输入输出形状/地址是固定的，replay 复用同一批缓冲。")
    print("  - 不能在图里有动态控制流或 host 回调；形状变化要重新录制。")
    print("  - torch 常用于推理（vLLM/TensorRT-LLM）和 torch.compile(reduce-overhead)。")


if __name__ == "__main__":
    main()
