#!/bin/bash

opt -load  ${LLVM_INSTALL_ROOT}/lib/LLVMbfc.so -bfc gramschmidt-cuda-nvptx64-nvidia-cuda-sm_60.bc #>static_out.txt 2>&1
opt -load  ${LLVM_INSTALL_ROOT}/lib/LLVMbfc.so -bfc gramschmidt.bc #>static_out.txt 2>&1

