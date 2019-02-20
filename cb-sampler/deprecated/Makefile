#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I$(CUDA_ROOT)/extras/CUPTI/include -I$(LIBUNWIND_INSTALL)/include
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
        LIBS= -lcuda -L $(CUDA_ROOT)/extras/CUPTI/lib64 -L $(LIBUNWIND_INSTALL)/lib -lcupti -lunwind -lunwind-x86_64
    endif
    OBJ = o
endif

callback_stacktrace: callback_stacktrace.$(OBJ)
	nvcc -o $@ callback_stacktrace.$(OBJ) $(LIBS) $(FLAGS)

callback_stacktrace.$(OBJ): callback_stacktrace.cu
	nvcc -c $(INCLUDES) $< $(FLAGS)

run: callback_stacktrace
	./$<

clean:
	rm -f callback_stacktrace callback_stacktrace.$(OBJ)
