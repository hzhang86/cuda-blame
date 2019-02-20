#!/bin/bash

# step1 compile & build
nvcc gesummv.cu -gencode=arch=compute_60,code=sm_60 -g -G -o gesummv -I${CUDA_ROOT}/include -I${CUDA_ROOT}/extras/CUPTI/include -I${LIBUNWIND_INSTALL}/include -I${HOME}/cuda-blame/cb-sampler -L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/extras/CUPTI/lib64 -L${LIBUNWIND_INSTALL}/lib -L${HOME}/cuda-blame/cb-sampler -L${CUDA_ROOT}/lib64/stubs -linst_sampling -lcupti -lunwind -lunwind-x86_64 -lcudadevrt -lcuda -lcudart_static -lrt -lpthread -ldl 

# step2 get bitcode
clang++ --cuda-gpu-arch=sm_60 --cuda-path=${CUDA_ROOT} -I${CUDA_ROOT}/include -I$CUDA_ROOT/extras/CUPTI/include -I${LIBUNWIND_INSTALL}/include -I${HOME}/cuda-blame/cb-sampler -g -c -emit-llvm gesummv.cu

# build the original program (without sampling instrumented)
#nvcc gesummv-orig.cu -gencode=arch=compute_60,code=sm_60 -g -G -o gesummv-orig -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib64/stubs -lcudadevrt -lcuda -lcudart_static -lrt -lpthread -ldl 
#nvcc gesummv-orig.cu -gencode=arch=compute_60,code=sm_60 -O3 -o gesummv-orig  
#nvcc gesummv-mod.cu -gencode=arch=compute_60,code=sm_60 -g -G -o gesummv-mod -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib64/stubs -lcudadevrt -lcuda -lcudart_static -lrt -lpthread -ldl 
#nvcc gesummv-mod.cu -gencode=arch=compute_60,code=sm_60 -O3 -o gesummv-mod 

