import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

def process_data_bal(test_size, n_splits, label): 
	''' 
	process_data_bal will create train, test, validation folds for all classes with balanced sample sizes up to n=750 and save them to the folder the script is run in
	requires path to final, feature table outputted from GDD Classifier R Script

	input:
		test_size = integer representing proportion out of 100 which should be used to make the testing set (default = 20, meaning test size is .2 of overall dataset)
		n_splits = integer, number of ensembles to create
		label = string input from .sh file representing specific label provided for each run ('' = normal, 'bal' = balanced, can define more)
	output:
		x_train_folds, y_train_folds  = list of np.arrays representing each ensemble training set for feature tables (x_train_folds) and labels (y_train_folds)
		x_test_folds, y_test_folds  = list of np.arrays representing each ensemble validation set for feature tables (x_test_folds) and labels (y_test_folds)
	'''
	data = pd.read_csv('/home/darmofam/morris/classifier/feature_table_all_cases_sigs.tsv', sep='\t') 
	#Removing those not annotated as a training sample (i.e. other, repeat patient samples, low purity), sarcoma
	data = data[data.Classification_Category == 'train']
	data = data[data.Cancer_Type != 'Sarcoma.NOS']
	labels = data.Cancer_Type
	ctypes = set(labels)
	#print(ctypes) #should be a list of 41 cancer type labels
	#print(len(ctypes)) #should be 41
	#split data into train and test before balancing
	data_train_labelled, data_test_labelled, labels_train_labelled, labels_test_labelled = train_test_split(data, labels, test_size=test_size/100, random_state = 0)
	#data drops the following labels
	data = data.drop(['SAMPLE_ID', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'SAMPLE_TYPE', 'PRIMARY_SITE', 'METASTATIC_SITE', 'Cancer_Type', 'Classification_Category'], axis=1)
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size/100, random_state = 0)
	#balance by oversampling to n = 750 for any types that dont have enough in training
	sampling_dic = {}
	for i in ctypes:
		sampling_dic[i] = max(list(labels_train).count(i), 750)
	ros = RandomOverSampler(random_state=42, sampling_strategy = sampling_dic) 
	data_train_bal, labels_train_bal = ros.fit_resample(data_train, labels_train)
	labels_train_bal.value_counts().to_csv('train_N_bal' + label + '.csv') #saves value counts
	#saving data tables -> 
	data_train_bal.to_csv('ft_train_bal' + label + '.csv', header = True, index = True)
	labels_train_bal.to_csv('labels_train_bal' + label + '.csv', header = True, index = True)
	data_test.to_csv('ft_test_bal' + label + '.csv', header = True, index = True)
	labels_test.to_csv('labels_test_bal' + label + '.csv', header = True, index = True)
	#now split into ensemble folds and encode tumor type labels
	sss = StratifiedShuffleSplit(n_splits=n_splits, random_state=0)
	sss.get_n_splits(data_train_bal, labels_train_bal)
	encoder = LabelEncoder()
	x_train_folds, x_test_folds= [], []
	y_train_folds, y_test_folds= [], []
	for train_index, test_index in sss.split(data_train_bal, labels_train_bal):
		#for each split, append np.arrays for training and testing feature tables, and encoded labels
		x_train, x_test = np.array(data_train_bal.iloc[train_index]), np.array(data_train_bal.iloc[test_index])
		y_train, y_test = np.array(labels_train_bal.iloc[train_index]), np.array(labels_train_bal.iloc[test_index]) 
		y_train = encoder.fit_transform(y_train)
		y_test = encoder.fit_transform(y_test)
		x_train_folds.append(x_train)
		x_test_folds.append(x_test)
		y_train_folds.append(y_train)
		y_test_folds.append(y_test)
	return x_train_folds, x_test_folds, y_train_folds, y_test_folds

test_size = 20
n_splits = 10
if len(sys.argv) > 1:
	label = '_' + sys.argv[1]
	print(label)
else:
	label = ''

x_train_folds, x_test_folds, y_train_folds, y_test_folds = process_data_bal(test_size, n_splits, label)
