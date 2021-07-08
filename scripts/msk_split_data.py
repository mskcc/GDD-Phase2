

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

def process_data_orig(test_size, n_splits, label, featureTable, outputDir, save_label):
	'''
	process_data_orig will create train, test, validation folds for all classes and save them to the folder the script is run in
	requires path to final, feature table outputted from GDD Classifier R Script

	input:
		test_size = integer representing proportion out of 100 which should be used to make the testing set (default = 20, meaning test size is .2 of overall dataset)
		n_splits = integer, number of ensembles to create
		label = string input from .sh file representing specific label provided for each run ('' = normal, 'bal' = balanced, can define more)
		save_label = Boolean for if you want to save the labelled versions of feature tables (X) and labels (y)
	output:
		x_train_folds, y_train_folds  = list of np.arrays representing each ensemble training set for feature tables (x_train_folds) and labels (y_train_folds)
		x_test_folds, y_test_folds  = list of np.arrays representing each ensemble validation set for feature tables (x_test_folds) and labels (y_test_folds)
	'''
	data = pd.read_csv(featureTable, sep='\t') #may need to change path to feature table
	#Removing those not annotated as a training sample (i.e. other, repeat patient samples, low purity), sarcoma
	data = data[data.Classification_Category == 'train']
	data = data[data.Cancer_Type != 'Sarcoma.NOS']
	labels = data.Cancer_Type
	ctypes = set(labels)
	#print(ctypes) #should be a list of 41 cancer type labels
	#print(len(ctypes)) #should be 41
	#splitting data to appropriate test size and saving value counts of each
	data_train_labelled, data_test_labelled, labels_train_labelled, labels_test_labelled = train_test_split(data, labels, test_size=test_size/100, random_state = 0)
	data_train_labelled.Cancer_Type.value_counts().to_csv(outputDir + 'train_N' + label + '.csv')
	if save_label: #only if you are saving the datasets with the cancer type labels
		data_train_labelled, data_test_labelled, labels_train_labelled, labels_test_labelled = train_test_split(data, labels, test_size=test_size/100, random_state = 0)
		data_train_labelled.to_csv(outputDir + 'ft_train_labelled' + label + '.csv', header = True, index = True)
		labels_train_labelled.to_csv(outputDir + 'labels_train_labelled' + label + '.csv', header = True, index = True)
		data_test_labelled.to_csv(outputDir + 'ft_test_labelled' + label + '.csv', header = True, index = True)
		labels_test_labelled.to_csv(outputDir + 'labels_test_labelled' + label + '.csv', header = True, index = True)
		print('done')
	#data drops the following labels
	data = data.drop(['SAMPLE_ID', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'SAMPLE_TYPE', 'PRIMARY_SITE', 'METASTATIC_SITE', 'Cancer_Type', 'Classification_Category'], axis=1)
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size/100, random_state = 0)
	#saving data tables ->
	data_train.to_csv(outputDir + 'ft_train' + label + '.csv', header = True, index = True)
	labels_train.to_csv(outputDir + 'labels_train' + label + '.csv', header = True, index = True)
	data_test.to_csv(outputDir + 'ft_test' + label + '.csv', header = True, index = True)
	labels_test.to_csv(outputDir + 'labels_test' + label + '.csv', header = True, index = True)
	#now split into ensemble folds and encode tumor type labels
	sss = StratifiedShuffleSplit(n_splits=n_splits, random_state=0)
	sss.get_n_splits(data_train, labels_train)
	encoder = LabelEncoder()
	x_train_folds, x_test_folds= [], []
	y_train_folds, y_test_folds= [], []
	for train_index, test_index in sss.split(data_train, labels_train):
		#for each split, append np.arrays for training and testing feature tables, and encoded labels
		x_train, x_test = np.array(data_train.iloc[train_index]), np.array(data_train.iloc[test_index])
		y_train, y_test = np.array(labels_train.iloc[train_index]), np.array(labels_train.iloc[test_index])
		y_train = encoder.fit_transform(y_train)
		y_test = encoder.fit_transform(y_test)
		x_train_folds.append(x_train)
		x_test_folds.append(x_test)
		y_train_folds.append(y_train)
		y_test_folds.append(y_test)
	return x_train_folds, x_test_folds, y_train_folds, y_test_folds



# Create the parser
parser = argparse.ArgumentParser(
    description='This program reads the list of command line arguments'
)

# Add Arguments
parser.add_argument('--featureTable', '-ft', type=str,
                    help='feature table file path')


parser.add_argument('--test_size', '-ts', type=int,
                    help='percentage split for test data')


parser.add_argument('--nsplits', '-ns', type=int,
                    help='number of splits for cross validation')

parser.add_argument('--outputDir', '-od', type=str, help='path of the output directory where results for this step will be saved')

# Execute the parse_args() method
args = parser.parse_args()


featureTable=args.featureTable
ft1=args.featureTable
test_size = args.test_size
n_splits = args.nsplits
label =''
outputDir=args.outputDir

# if len(sys.argv) > 1:
# 	label = '_' + sys.argv[1]
# 	print(label)
# else:
# 	label = ''

# print(featureTable)
#print(outputDir)


x_train_folds, x_test_folds, y_train_folds, y_test_folds = process_data_orig(test_size, n_splits, label, featureTable, outputDir, save_label=False)
