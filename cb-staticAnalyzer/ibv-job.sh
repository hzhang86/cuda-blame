#!/bin/bash
#SBATCH -t 0:15:0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --output=./job.output

echo "Start running make for BFC pass in llvm!"
make REQUIRES_RTTI=1 ENABLE_CXX11=1
