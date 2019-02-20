#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I$(CUDA_ROOT)/extras/CUPTI/include -I$(LIBUNWIND_INSTALL)/include -I$(HOME)/cuda-blame/cb-sampler
FLAGS = -arch=sm_52 -G -g

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):../../libWin32:../../libx64
    LIBS= -lcuda -L ../../libWin32 -L ../../libx64 -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):../../lib
        LIBS= -Xlinker -framework -Xlinker cuda -L ../../lib -lcupti
    else # we only modified here for linux redhat/ubuntu
        LIBS= -lcuda -L $(CUDA_ROOT)/extras/CUPTI/lib64 -L $(LIBUNWIND_INSTALL)/lib -L $(HOME)/cuda-blame/cb-sampler -lcupti -lunwind -lunwind-x86_64 -linst_sampling
    endif
    OBJ = o
endif

vecAddSub: vecAddSub.$(OBJ)
	nvcc -o $@ vecAddSub.$(OBJ) $(LIBS) 

vecAddSub.$(OBJ): vecAddSub.cu
	nvcc -c $(INCLUDES) $< $(FLAGS)

run: vecAddSub
	./$<

clean:
	rm -f vecAddSub vecAddSub.$(OBJ)
