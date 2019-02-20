#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:m40:1
#SBATCH --output=./job.output
#SBATCH --mem 10gb

#LD_PRELOAD=/lustre/hzhang86/miscellaneous/cudaCode/cuptiActivity/cuptiActivity.so ./lulesh -u sedov15oct.lmesh
#ENABLE_CUDA_SAMP=1 srun ./lulesh -u sedov15oct.lmesh #-s 15
./vecAddSub
#addParser vecAddSub vecAddSub-cuda-nvptx64-nvidia-cuda-sm_35.bc.calls CPU.txt GPU.txt
