#!/bin/bash
# build .so file, used as LD_PRELOAD
#nvcc -g -Xcompiler -fPIC -I/cell_root/software/cuda/7.5.18/sys/extras/CUPTI/include -c cuptiActivity.cu
#nvcc -shared -o cuptiActivity.so cuptiActivity.o -L/cell_root/software/cuda/7.5.18/sys/lib64 -L/cell_root/software/cuda/7.5.18/sys/extras/CUPTI/lib64 -L/cell_root/software/cuda/7.5.18/sys/lib64/stubs -lcuda -lcupti -lcudart_static -ldl -lrt 

# build a library, used as LD_LIBRARY_PATH
#nvcc -Xcompiler -fPIC -I$CUDA_ROOT/extras/CUPTI/include -I$LIBUNWIND_INSTALL/include -c inst_sampling.cu -arch=sm_52 -G -g
gcc -fPIC -I$CUDA_ROOT/include -I$CUDA_ROOT/extras/CUPTI/include -I$LIBUNWIND_INSTALL/include -c -g inst_sampling.cpp 
echo "Done compiling"
gcc -shared -Wl,-soname,libinst_sampling.so -o libinst_sampling.so inst_sampling.o -L$CUDA_ROOT/lib64 -L$CUDA_ROOT/extras/CUPTI/lib64 -L$CUDA_ROOT/lib64/stubs -lcuda -lcupti -lcudart_static -ldl -lrt
echo "Done linking"
