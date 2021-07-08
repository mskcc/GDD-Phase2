#!/usr/bin/env bash

# source ~/.bashrc
# module load cuda/10.1
# conda activate vir-env



inputDir='../Data/output/step1/'
outputDir='../Data/output/step2/'

mkdir -p $outputDir 2>/dev/null

testSize=20
n_splits=10
split_fold=1


cmd="python msk_clfin.py -ts $testSize -ns $n_splits -id $inputDir -od $outputDir -sf $split_fold"
#python msk_clfin.py "$((${LSB_JOBINDEX}-1))


#cmd="python msk_split_data.py -ft ${featureTable} -ts $testSize -ns $n_splits -od $outputDir"



date
echo "Job Starting"
echo $cmd

eval $cmd
echo
date
echo "All done"
