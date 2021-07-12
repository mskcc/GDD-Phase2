#!/usr/bin/env bash


# source ~/.bashrc
# module load cuda/10.1
# conda activate vir-env
# python msk_clfin.py "$((${LSB_JOBINDEX}-1))"

step=2
rLSF="rusage[mem=24]"
nLSF=4
mLSF="lt-gpu"
qLSF="gpuqueue"
gpuLSF="num=1:j_exclusive=yes:mode=shared"

inputDir='../Data/output/step1/'
outputDir=../Data/output/step$step/
logDir=../Data/Log/step$step

mkdir -p $outputDir 2>/dev/null
mkdir -p $logDir 2>/dev/null

testSize=20
n_splits=10
# n_splits=2

module load cuda/10.1
source /home/sumans/miniconda3/bin/activate
conda activate gddP2

for ((split_fold=0; split_fold<$n_splits;split_fold++)); do
  #split_fold=$split
  #echo $split_fold

  cmd="bsub \
      -W 72:00 \
      -o ${logDir}/step${step}_split_${split_fold}.out \
      -eo ${logDir}/step${step}_split_${split_fold}.stderr \
      -m \"$mLSF\" \
      -q $qLSF \
      -n $nLSF \
      -R \"$rLSF\" \
      -gpu \"$gpuLSF\" \
      -J gddP2_step${step}_split_${split_fold} \
      python msk_clfin.py -ts $testSize -ns $n_splits -id $inputDir -od $outputDir -sf $split_fold"




  date
  echo "Job Starting for split_fold = $split_fold"
  echo
  echo $cmd

  eval $cmd
  echo
  date
  echo "All done"
  echo
  echo



done
