#!/usr/bin/env bash

#module load cuda/10.1
#conda activate vir-env


inputDataDir='../Data/'
featureTable=${inputDataDir}/FeatureTable/feature_table_all_cases_sigs.tsv

outputDir='../Data/output/step1/'

mkdir -p $outputDir 2>/dev/null

testSize=20
n_splits=10

#
# echo
# echo $inputDataDir
#
# echo $featureTable

cmd="python msk_split_data.py -ft ${featureTable} -ts $testSize -ns $n_splits -od $outputDir"



date
echo "Job Starting"
echo $cmd

eval $cmd
echo
date
echo "All done"
