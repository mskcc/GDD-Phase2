#!/usr/bin/env bash

step=1
rLSF="rusage[mem=24]"
nLSF=1
# mLSF="ls-gpu"
mLSF="lt-gpu"
qLSF="gpuqueue -n 1 -gpu \"num=1:mps=yes\""
gpuLSF="num=1:j_exclusive=yes:mode=shared"

inputDataDir='../Data/'
featureTable=${inputDataDir}/FeatureTable/feature_table_all_cases_sigs.tsv

outputDir=../Data/output/step${step}/
logDir=../Data/Log/step$step

mkdir -p $outputDir 2>/dev/null
mkdir -p $logDir 2>/dev/null

testSize=20
n_splits=10

# source ~/.bashrc
module load cuda/10.1
source /home/sumans/miniconda3/bin/activate
conda activate gddP2
#conda activate vir-env
#python msk_split_data.py

cmd="bsub \
    -W 72:00 \
    -o ${logDir}/step${step}.out \
    -eo ${logDir}/step${step}.stderr \
    -m \"$mLSF\" \
    -q $qLSF \
    -n $nLSF \
    -R \"$rLSF\" \
    -gpu \"$gpuLSF\" \
    -J gddP2_step${step} \
    python msk_split_data.py -ft ${featureTable} -ts $testSize -ns $n_splits -od $outputDir"

date
echo "Job Starting"
echo $cmd

eval $cmd
echo
date
echo "All done"
