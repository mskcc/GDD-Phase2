#!/usr/bin/env bash


inputDataDir='../Data/'
featureTable=${inputDataDir}/FeatureTable/feature_table_all_cases_sigs.tsv

outputDir='../Data/output/step1/'

mkdir -p $outputDir 2>/dev/null

testSize=20
n_splits=10

# source ~/.bashrc
# module load cuda/10.1
# conda activate vir-env
#python msk_split_data.py

cmd="bsub \
    -W 72.00 \
    -o split_data.out \
    -eo split_data.stderr \
    -m ly-gpu \
    -q gpuqueue -n 1 -gpu num=1:mps=yes \
    -n 1 -R rusage[mem=24] \
    -gpu num=1:j_exclusive=yes:mode=shared \
    -J split_data \
    python msk_split_data.py -ft ${featureTable} -ts $testSize -ns $n_splits -od $outputDir"

date
echo "Job Starting"
echo $cmd

#eval $cmd
echo
date
echo "All done"
