// 04_events_timing.cu —— 用事件（event）精确计时
//
// 学习目标：
//   1. 事件（cudaEvent_t）是"流上的一个标记/时间戳"。
//   2. 因为内核启动是异步的，主机时钟测不到真实执行时间，必须用事件。
//   3. 事件记录了 GPU 时间线（device timeline）上的时刻，
//      cudaEventElapsedTime 读出两个事件之间的耗时。
//   4. 读耗时前必须 cudaEventSynchronize 保证事件已完成。
//   5. 不需要计时的事件用 cudaEventDisableTiming 省掉计时开销。
//
// 涉及 API：cudaEventCreate / cudaEventCreateWithFlags /
//           cudaEventRecord / cudaEventSynchronize / cudaEventElapsedTime
#include "common.h"
#include <chrono>

int main() {
    const int GRID = 512, BLK = 256;
    float* d;
    CHECK(cudaMalloc(&d, (size_t)GRID * BLK * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    std::printf("===== 内核耗时：主机时钟 vs 事件计时 =====\n");

    // 错误示范：主机时钟测内核（因为异步，只测到"排队"，几乎为 0）
    int ITERS = 2000;
    auto t0 = std::chrono::steady_clock::now();
    spin_kernel<<<GRID, BLK>>>(d, ITERS);
    auto t1 = std::chrono::steady_clock::now();
    double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 正确做法：在流里记录两个事件，再用 cudaEventElapsedTime
    CHECK(cudaEventRecord(start));
    spin_kernel<<<GRID, BLK>>>(d, ITERS);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float event_ms = elapsed_ms(start, stop);

    std::printf("  主机时钟(错误): %.4f ms   <- 只测到排队，不是执行\n", host_ms);
    std::printf("  事件计时(正确): %.3f ms   <- 真实执行时间\n", event_ms);

    // ---- 读耗时前必须同步 -----------------------------------------------------
    std::printf("\n===== cudaEventElapsedTime 前必须等事件完成 =====\n");
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    spin_kernel<<<GRID, BLK>>>(d, ITERS * 30);   // 故意跑久一点，确保还没结束
    CHECK(cudaEventRecord(b));
    float ms = -1.0f;
    cudaError_t err = cudaEventElapsedTime(&ms, a, b);
    std::printf("  未同步就查询: err=%s (%s)\n",
                cudaGetErrorName(err), cudaGetErrorString(err));
    CHECK(cudaEventSynchronize(b));
    std::printf("  同步后再查询: %.3f ms\n", elapsed_ms(a, b));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));

    // ---- cudaEventDisableTiming ----------------------------------------------
    std::printf("\n===== cudaEventDisableTiming =====\n");
    cudaEvent_t e_sync;
    CHECK(cudaEventCreateWithFlags(&e_sync, cudaEventDisableTiming));
    std::printf("  禁用计时的事件：仅作同步点用，创建/记录开销更低。\n");
    std::printf("  但无法对 cudaEventDisableTiming 的事件调用 cudaEventElapsedTime。\n");
    CHECK(cudaEventDestroy(e_sync));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d));
    return 0;
}
