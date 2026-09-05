// 06_stream_semantics.cu —— 跨流依赖 & 默认流/非阻塞流语义
//
// 学习目标：
//   1. cudaStreamWaitEvent(stream, event)：让一个流等待另一个流里的"某个点"，
//      这是建立"精确"跨流依赖的手段（比 cudaDeviceSynchronize 细得多）。
//   2. 遗留默认流（stream 0）会与其它"阻塞"流隐式同步，可能悄悄串行化你的程序。
//   3. cudaStreamNonBlocking 标志创建的非阻塞流，不参与遗留默认流的隐式同步。
//
// 涉及 API：cudaStreamWaitEvent / cudaStreamCreateWithFlags / cudaStreamNonBlocking
#include "common.h"

int main() {
    const int GRID = 512, BLK = 256;
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, (size_t)GRID * BLK * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)GRID * BLK * sizeof(float)));

    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));

    cudaEvent_t ev1, start, stop;
    CHECK(cudaEventCreate(&ev1));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    std::printf("===== cudaStreamWaitEvent：让 s2 等 s1 的某个\"点\" =====\n");

    // s1 上先跑一个长任务，结束时在 s1 里记录事件 ev1
    spin_kernel<<<GRID, BLK, 0, s1>>>(d_in, 3000);
    CHECK(cudaEventRecord(ev1, s1));

    // s2 的任务依赖 s1 的结果 -> 让 s2 等 ev1 这个"点"
    CHECK(cudaStreamWaitEvent(s2, ev1, 0));

    // 之后 s1 继续跑第二个长任务（与 s2 无关）
    CHECK(cudaEventRecord(start));
    spin_kernel<<<GRID, BLK, 0, s1>>>(d_in, 3000);   // s1 继续
    spin_kernel<<<GRID, BLK, 0, s2>>>(d_out, 200);   // s2 只等 ev1，不等 s1 的第二个任务
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = elapsed_ms(start, stop);

    std::printf("  从 s2 提交到完成耗时 %.3f ms\n", ms);
    std::printf("  语义：s2 只等 s1 的第一个任务(ev1)，不等 s1 之后的任务。\n");
    std::printf("  -> 这就是\"精确到点\"的跨流依赖，比同步整个设备细得多。\n");

    // ---- 非阻塞流 -------------------------------------------------------------
    std::printf("\n===== cudaStreamNonBlocking 非阻塞流 =====\n");
    cudaStream_t nb;
    CHECK(cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking));
    std::printf("  用 cudaStreamNonBlocking 创建的流，不参与遗留默认流的隐式同步。\n");
    std::printf("  （普通 cudaStreamCreate 出的流是\"阻塞\"流，会与默认流隐式同步。）\n");

    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaStreamDestroy(nb));
    CHECK(cudaEventDestroy(ev1));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    return 0;
}
