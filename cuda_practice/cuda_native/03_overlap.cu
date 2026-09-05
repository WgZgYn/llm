// 03_overlap.cu —— 异步拷贝与内核计算的重叠（overlap）
//
// 学习目标：
//   拷贝走 GPU 的 copy engine（DMA），计算走 SM。二者是不同硬件单元，
//   因此可以"重叠"：一边拷贝下一块数据，一边计算上一块数据。
//   但前提是：两者必须放在不同的流上，且拷贝用 pinned 内存 + cudaMemcpyAsync。
//   若放在同一个流，流是 FIFO 队列，会强制串行。
//
// 涉及 API：cudaStreamCreate / cudaMemcpyAsync / cudaEventRecord /
//           cudaEventElapsedTime / cudaMallocHost
#include "common.h"

int main() {
    const size_t N = 1 << 26;               // 256 MB
    const size_t bytes = N * sizeof(float);
    const int GRID = 512, BLK = 256;
    const int ITERS = 2000;                 // 计算负载，稍后按实测校准

    // 主机 pinned 缓冲（异步拷贝前提）与两个设备缓冲
    // 注意：计算写 d_out，拷贝写 d —— 二者无数据依赖，重叠才是"合法"的。
    float *h, *d, *d_out;
    CHECK(cudaMallocHost(&h, bytes));
    CHECK(cudaMalloc(&d, bytes));
    CHECK(cudaMalloc(&d_out, (size_t)GRID * BLK * sizeof(float)));

    cudaStream_t s_copy, s_compute;
    CHECK(cudaStreamCreate(&s_copy));
    CHECK(cudaStreamCreate(&s_compute));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // ---- 校准：单独测拷贝耗时、计算耗时 --------------------------------------
    CHECK(cudaEventRecord(start, s_copy));
    CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s_copy));
    CHECK(cudaEventRecord(stop, s_copy));
    CHECK(cudaEventSynchronize(stop));
    float t_copy = elapsed_ms(start, stop);

    CHECK(cudaEventRecord(start, s_compute));
    spin_kernel<<<GRID, BLK, 0, s_compute>>>(d_out, ITERS);
    CHECK(cudaEventRecord(stop, s_compute));
    CHECK(cudaEventSynchronize(stop));
    float t_compute = elapsed_ms(start, stop);

    std::printf("单独执行：拷贝 %.3f ms，计算 %.3f ms\n", t_copy, t_compute);
    std::printf("若完全串行，总耗时 ≈ %.3f ms\n\n", t_copy + t_compute);

    // ---- 情况 A：都在同一个流 -> 串行 ----------------------------------------
    cudaStream_t s_single;
    CHECK(cudaStreamCreate(&s_single));
    CHECK(cudaEventRecord(start, s_single));
    CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s_single));
    spin_kernel<<<GRID, BLK, 0, s_single>>>(d_out, ITERS);
    CHECK(cudaEventRecord(stop, s_single));
    CHECK(cudaEventSynchronize(stop));
    float t_serial = elapsed_ms(start, stop);

    // ---- 情况 B：拷贝/计算分到两个流 -> 重叠 -----------------------------------
    CHECK(cudaEventRecord(start));
    CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s_copy));
    spin_kernel<<<GRID, BLK, 0, s_compute>>>(d_out, ITERS);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float t_overlap = elapsed_ms(start, stop);

    float lower_bound = (t_copy > t_compute) ? t_copy : t_compute;
    std::printf("===== 结果 =====\n");
    std::printf("  同流串行      : %8.3f ms\n", t_serial);
    std::printf("  双流重叠      : %8.3f ms\n", t_overlap);
    std::printf("  理论重叠下界  : %8.3f ms  (= max(拷贝, 计算))\n", lower_bound);
    if (t_serial > 0)
        std::printf("  节省比例      : %.1f%%\n",
                    100.0 * (t_serial - t_overlap) / t_serial);

    CHECK(cudaStreamDestroy(s_single));
    CHECK(cudaStreamDestroy(s_copy));
    CHECK(cudaStreamDestroy(s_compute));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFreeHost(h));
    CHECK(cudaFree(d));
    CHECK(cudaFree(d_out));
    return 0;
}
