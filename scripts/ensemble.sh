#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 72:00
#
# Set output file
#BSUB -o  ensemble.out
#
# Set error file
#BSUB -eo ensemble.stderr
#
# Specify node group
#BSUB -m "ly-gpu"
#BSUB -q gpuqueue -n 1 -gpu "num=1:mps=yes"
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=10]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "ensemble"
source ~/.bashrc
module load cuda/10.1
conda activate vir-env
python msk_ensemble.py 

