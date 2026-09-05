// 02_async_memcpy.cu —— 同步拷贝 vs 异步拷贝，以及 pageable vs pinned 内存
//
// 学习目标：
//   1. cudaMemcpy 是"同步"的：主机阻塞到拷贝完成才返回。
//   2. cudaMemcpyAsync 是"异步"的：把拷贝排进流后立刻返回。
//   3. 但异步拷贝要真正异步，主机内存必须是 pinned（page-locked）。
//      普通 pageable 内存下，驱动会退化为"同步 + 中转（staging buffer）"，
//      既慢又失去异步性。
//
// 涉及 API：cudaMemcpy / cudaMemcpyAsync / cudaMallocHost / cudaFreeHost
#include "common.h"
#include <cstdlib>

int main() {
    const size_t N = 1 << 26;            // 64M 个 float
    const size_t bytes = N * sizeof(float);   // 256 MB

    // 设备缓冲
    float* d;
    CHECK(cudaMalloc(&d, bytes));

    // 主机缓冲（pageable，普通 malloc）
    float* h_pageable = (float*)std::malloc(bytes);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    std::printf("===== H2D 拷贝 256 MB：pageable/pinned × 同步/异步 =====\n\n");

    // 同步拷贝（pageable）
    CHECK(cudaEventRecord(start));
    CHECK(cudaMemcpy(d, h_pageable, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float t_sync = elapsed_ms(start, stop);
    std::printf("  cudaMemcpy        (pageable) : %8.3f ms   %7.2f GB/s\n",
                t_sync, gbps(bytes, t_sync));

    // 异步拷贝（pageable）—— 表面异步，实际会被驱动同步化
    CHECK(cudaEventRecord(start));
    CHECK(cudaMemcpyAsync(d, h_pageable, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float t_async_pageable = elapsed_ms(start, stop);
    std::printf("  cudaMemcpyAsync   (pageable) : %8.3f ms   %7.2f GB/s\n",
                t_async_pageable, gbps(bytes, t_async_pageable));

    // 异步拷贝（pinned）—— 真正异步 + 带宽更高
    float* h_pinned;
    CHECK(cudaMallocHost(&h_pinned, bytes));
    CHECK(cudaEventRecord(start));
    CHECK(cudaMemcpyAsync(d, h_pinned, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float t_async_pinned = elapsed_ms(start, stop);
    std::printf("  cudaMemcpyAsync   (pinned)   : %8.3f ms   %7.2f GB/s\n",
                t_async_pinned, gbps(bytes, t_async_pinned));

    std::printf("\n  结论要点：\n");
    std::printf("  - pinned 内存让 cudaMemcpyAsync 真正异步，带宽也更高（免去 staging 中转）。\n");
    std::printf("  - pageable 内存下 cudaMemcpyAsync 退化为同步，异步收益消失。\n");
    std::printf("  - 所以做异步流水线/重叠时，主机侧缓冲必须用 cudaMallocHost 分配。\n");

    std::free(h_pageable);
    CHECK(cudaFreeHost(h_pinned));
    CHECK(cudaFree(d));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return 0;
}
