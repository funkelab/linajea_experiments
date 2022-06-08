#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH -t 4-00:00:00
#SBATCH -o %A-%a.out
#SBATCH -e %A-%a.err

# export CUDA_VISIBLE_DEVICES=0

python "$@"
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit 100
fi
