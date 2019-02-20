#!/bin/bash

# get the binary
#clang++ axpy.cu -g -o axpy --cuda-path=/cell_root/software/cuda/7.5.18/sys/ --cuda-gpu-arch=sm_35 -L/cell_root/software/cuda/7.5.18/sys/lib64 -lcudart_static -ldl -lrt -pthread
# get the bitcode format
clang++ vecAddSub.cu -c -g -emit-llvm -I${CUDA_ROOT}/extras/CUPTI/include -I${LIBUNWIND_INSTALL}/include -I${CUDA_BLAME_ROOT}/cb-sampler --cuda-path=${CUDA_ROOT} --cuda-gpu-arch=sm_52
