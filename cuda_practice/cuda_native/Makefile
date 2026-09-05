# 在 Linux 上编译原生 CUDA C++ 示例（本机 Windows 无 MSVC 时用这个）
# 用法：make            （默认按 GPU 自动探测架构）
#       make ARCH=sm_89
NVCC      ?= nvcc
ARCH      ?= $(shell nvcc --list-gpu-arch | tail -1)   # 取最后一个支持的 compute_xx
# 注意：--list-gpu-arch 输出形如 compute_89，需转成 sm_89
ARCH      := $(subst compute_,sm_,$(ARCH))
CXXFLAGS  ?= -O3 -std=c++11
SRCS      := $(wildcard *.cu)
BINS      := $(patsubst %.cu,%,$(SRCS))

all: $(BINS)

%: %.cu common.h
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) -I. -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
