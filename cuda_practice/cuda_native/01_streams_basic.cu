// 01_streams_basic.cu —— 流（stream）的基本概念
//
// 学习目标：
//   1. 内核启动是"异步"的：主机把任务丢进队列后立刻返回，不等待执行完成。
//   2. 流（cudaStream_t）是一个 FIFO 队列，同一个流内的操作按提交顺序执行。
//   3. 不同流内的操作可以并发执行。
//
// 涉及 API：cudaStreamCreate / cudaStreamDestroy / cudaEventCreate /
//           cudaEventRecord / cudaEventSynchronize / cudaEventElapsedTime
#include "common.h"
#include <chrono>

int main() {
    const int GRID = 256, BLK = 256;   // 256*256 = 65536 个线程
    const int ITERS = 1000;            // spin 迭代次数，控制单 kernel 耗时

    float* d_out;
    CHECK(cudaMalloc(&d_out, (size_t)GRID * BLK * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // ---- 1) 内核启动是异步的 -------------------------------------------------
    std::printf("===== 1) 内核启动是异步的 =====\n");

    auto t0 = std::chrono::steady_clock::now();
    spin_kernel<<<GRID, BLK>>>(d_out, ITERS);   // 默认流 (stream 0)
    auto t1 = std::chrono::steady_clock::now();
    double launch_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  kernel 启动（把任务入队）仅用 %.4f ms —— 主机没等它算完\n",
                launch_ms);

    // 真正要等它算完，必须显式同步；用事件测量真实执行耗时
    CHECK(cudaEventRecord(start));            // 默认流
    spin_kernel<<<GRID, BLK>>>(d_out, ITERS);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    std::printf("  内核实际执行耗时: %.3f ms（事件计时）\n",
                elapsed_ms(start, stop));

    // ---- 2) 流是 FIFO 队列，流内按顺序执行 ------------------------------------
    std::printf("\n===== 2) 流内操作按顺序（FIFO）执行 =====\n");
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));

    CHECK(cudaEventRecord(start, s));
    spin_kernel<<<GRID, BLK, 0, s>>>(d_out, ITERS);
    spin_kernel<<<GRID, BLK, 0, s>>>(d_out, ITERS);
    CHECK(cudaEventRecord(stop, s));
    CHECK(cudaEventSynchronize(stop));
    float two_serial = elapsed_ms(start, stop);
    std::printf("  同一流上两个 kernel 总耗时: %.3f ms（≈ 2× 单 kernel，说明串行）\n",
                two_serial);

    // ---- 3) 不同流上的操作可以并发 --------------------------------------------
    std::printf("\n===== 3) 不同流上的操作可以并发 =====\n");
    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));

    CHECK(cudaEventRecord(start));
    spin_kernel<<<GRID, BLK, 0, s1>>>(d_out, ITERS);
    spin_kernel<<<GRID, BLK, 0, s2>>>(d_out, ITERS);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float two_parallel = elapsed_ms(start, stop);

    std::printf("  两个流上各一个 kernel 总耗时: %.3f ms（若≈单 kernel 耗时，说明并发）\n",
                two_parallel);
    std::printf("  -> 串行 %.3f ms  vs  并行 %.3f ms\n", two_serial, two_parallel);

    CHECK(cudaStreamDestroy(s));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_out));
    return 0;
}
