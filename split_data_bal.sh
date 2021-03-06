#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 72:00
#
# Set output file
#BSUB -o  split_data_bal.out
#
# Set error file
#BSUB -eo split_data_bal.stderr
#
# Specify node group
#BSUB -m "ly-gpu"
#BSUB -q gpuqueue -n 1 -gpu "num=1:mps=yes"
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=35]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "split_data_bal"
source ~/.bashrc
module load cuda/10.1
conda activate vir-env
python msk_split_data_bal.py

