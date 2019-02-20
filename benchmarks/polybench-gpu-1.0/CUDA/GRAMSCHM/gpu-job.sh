#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./job.output.csv

nvprof ./gramschmidt-orig
nvprof ./gramschmidt-mod
./gramschmidt
