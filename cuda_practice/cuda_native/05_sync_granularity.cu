// 05_sync_granularity.cu —— 同步的粒度与代价
//
// 学习目标：
//   CUDA 有三档同步粒度，越细代价越低、对并发破坏越小：
//     cudaDeviceSynchronize()   等整个设备所有流 —— 最粗、最贵
//     cudaStreamSynchronize(s)  等某一个流        —— 中等
//     cudaEventSynchronize(e)   等某一个"点"      —— 最细、最便宜
//   此外还有"非阻塞查询"（query）和"阻塞等待"（synchronize）之分。
//
// 涉及 API：cudaDeviceSynchronize / cudaStreamSynchronize /
//           cudaEventSynchronize / cudaStreamQuery / cudaEventQuery
#include "common.h"
#include <chrono>

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main() {
    const int N_ITER = 1000;

    // ---- 1) 空闲同步的开销：设备空闲时，同步调用本身要花多少 -------------------
    std::printf("===== 1) 空闲同步调用开销（%d 次平均）=====\n", N_ITER);
    {
        cudaStream_t s;
        cudaEvent_t e;
        CHECK(cudaStreamCreate(&s));
        CHECK(cudaEventCreate(&e));
        CHECK(cudaEventRecord(e, s));
        CHECK(cudaEventSynchronize(e));

        double t0 = now_ms();
        for (int i = 0; i < N_ITER; ++i) CHECK(cudaDeviceSynchronize());
        double t_dev = (now_ms() - t0) / N_ITER;

        t0 = now_ms();
        for (int i = 0; i < N_ITER; ++i) CHECK(cudaStreamSynchronize(s));
        double t_str = (now_ms() - t0) / N_ITER;

        t0 = now_ms();
        for (int i = 0; i < N_ITER; ++i) CHECK(cudaEventSynchronize(e));
        double t_evt = (now_ms() - t0) / N_ITER;

        std::printf("  cudaDeviceSynchronize  : %7.3f us/次\n", t_dev * 1000.0);
        std::printf("  cudaStreamSynchronize : %7.3f us/次\n", t_str * 1000.0);
        std::printf("  cudaEventSynchronize  : %7.3f us/次\n", t_evt * 1000.0);

        CHECK(cudaStreamDestroy(s));
        CHECK(cudaEventDestroy(e));
    }

    // ---- 2) 粒度语义：device 等所有流，stream 只等一个流 ------------------------
    std::printf("\n===== 2) 同步粒度：device(等所有) vs stream(等一个) =====\n");
    {
        float* d;
        CHECK(cudaMalloc(&d, (size_t)512 * 256 * sizeof(float)));
        cudaStream_t sA, sB;
        CHECK(cudaStreamCreate(&sA));
        CHECK(cudaStreamCreate(&sB));

        spin_kernel<<<512, 256, 0, sA>>>(d, 3000);   // 长任务
        spin_kernel<<<512, 256, 0, sB>>>(d, 200);    // 短任务

        double t0 = now_ms();
        CHECK(cudaStreamSynchronize(sB));
        double t_syncB = now_ms() - t0;
        std::printf("  cudaStreamSynchronize(sB) 用时 %6.3f ms（只等短任务，快）\n",
                    t_syncB);

        spin_kernel<<<512, 256, 0, sA>>>(d, 3000);   // 再放一个长任务
        t0 = now_ms();
        CHECK(cudaDeviceSynchronize());
        double t_devsync = now_ms() - t0;
        std::printf("  cudaDeviceSynchronize()   用时 %6.3f ms（等所有流，含长任务）\n",
                    t_devsync);

        CHECK(cudaStreamDestroy(sA));
        CHECK(cudaStreamDestroy(sB));
        CHECK(cudaFree(d));
    }

    // ---- 3) 阻塞同步 vs 轮询查询 ----------------------------------------------
    std::printf("\n===== 3) 阻塞同步 vs 轮询（query）=====\n");
    {
        float* d;
        CHECK(cudaMalloc(&d, (size_t)512 * 256 * sizeof(float)));
        cudaStream_t s;
        CHECK(cudaStreamCreate(&s));

        // 阻塞：主机干等
        spin_kernel<<<512, 256, 0, s>>>(d, 3000);
        double t0 = now_ms();
        CHECK(cudaStreamSynchronize(s));
        double t_block = now_ms() - t0;

        // 轮询：主机一边查一边可以做别的事（这里用计数器模拟"别的活"）
        spin_kernel<<<512, 256, 0, s>>>(d, 3000);
        long polls = 0;
        t0 = now_ms();
        cudaError_t e;
        do {
            ++polls;
            e = cudaStreamQuery(s);   // 非阻塞，立刻返回 cudaSuccess / cudaErrorNotReady
        } while (e == cudaErrorNotReady);
        double t_poll = now_ms() - t0;

        std::printf("  阻塞 cudaStreamSynchronize : %6.3f ms（干等，不占 CPU）\n",
                    t_block);
        std::printf("  轮询 cudaStreamQuery       : %6.3f ms（查了 %ld 次，期间可干别的）\n",
                    t_poll, polls);

        CHECK(cudaStreamDestroy(s));
        CHECK(cudaFree(d));
    }
    return 0;
}
