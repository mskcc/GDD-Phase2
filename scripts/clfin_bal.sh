#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 72:00
#
# Set output file
#BSUB -o  fold.%I.bal.out
#
# Set error file
#BSUB -eo fold.%I.bal.stderr
#
# Specify node group
#BSUB -m "ly-gpu"
#BSUB -q gpuqueue
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=30]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "mskclfin_bal_fold[1-10]"
source ~/.bashrc
module load cuda/10.1
conda activate vir-env
python msk_clfin_bal.py "$((${LSB_JOBINDEX}-1))"
