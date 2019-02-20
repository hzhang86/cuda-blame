#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./job.output.csv

./gesummv-orig
./gesummv-mod
./gesummv

