
#!/usr/bin/env bash



dlModel='/Users/shalabhs/Projects/GDD/Project_GDDV2/Data/ensemble_classifier_update_bal.pt'

# inputFile='/Users/shalabhs/Projects/GDD/Project_GDDV2/GDD-Phase2/features_20220519160652.%f_77a49988.txt'
inputFile='single_test.csv'

outputFile='single_res_2.csv'

script='predict_single_ss.py'

cmd="python $script $dlModel $inputFile $outputFile"

echo $cmd

eval $cmd
