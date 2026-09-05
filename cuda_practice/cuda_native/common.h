// common.h —— 各示例共用的工具与测试内核
//
// 这里的工具刻意保持"原生 CUDA Runtime API"风格，方便对照 CUDA 官方文档。
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// 每个 CUDA 调用都用 CHECK 包一层：出错即打印文件/行号/错误码并退出。
// 学习阶段这样做能立刻定位是哪一次 API 调用失败。
#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t _err = (call);                                      \
        if (_err != cudaSuccess) {                                      \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d  %s (%s)\n",       \
                         __FILE__, __LINE__, cudaGetErrorName(_err),    \
                         cudaGetErrorString(_err));                     \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

// 两个已记录事件之间的毫秒数（cudaEventElapsedTime 的封装）。
// 注意：调用前必须保证两个事件都已"完成"，否则返回 cudaErrorNotReady。
inline float elapsed_ms(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

// bytes / ms -> GB/s
inline double gbps(double bytes, double ms) {
    if (ms <= 0.0) return 0.0;
    return bytes / (ms * 1e6);
}

// 字节数 -> 人类可读（B/KB/MB/GB/TB）
inline std::string human(double bytes) {
    const char* u[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0;
    double b = bytes;
    while (b >= 1024.0 && i < 4) { b /= 1024.0; ++i; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", b, u[i]);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// 纯计算内核：用 FMA 循环占满 SM 的算力，几乎不读写显存。
// 用于制造可控的"计算负载"，方便观察与异步拷贝（走 copy engine）的并发/重叠。
// iters 越大耗时越长。写入 out 是为了防止编译器把整个循环优化掉。
// ---------------------------------------------------------------------------
__global__ void spin_kernel(float* out, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)idx * 1e-6f;
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 1.000001f, 1.0f);
        x = fmaf(x, 0.999999f, -1.0f);
    }
    out[idx] = x;
}

// 内存带宽型内核：SAXPY  y[i] = a*x[i] + y[i]。
// 用于展示"受显存带宽约束"的负载——它和拷贝争同一份显存带宽，难以重叠。
__global__ void saxpy_kernel(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
