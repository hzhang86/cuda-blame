#!/bin/bash

opt -load ${LLVM_INSTALL_ROOT}/lib/LLVMbfc.so -bfc vecAddSub-cuda-nvptx64-nvidia-cuda-sm_52.bc
opt -load ${LLVM_INSTALL_ROOT}/lib/LLVMbfc.so -bfc vecAddSub.bc
