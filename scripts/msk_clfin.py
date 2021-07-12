
import os
import sys
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_objective_2D
from skopt.utils import use_named_args
#print('imports finished')

class MyDataset(Dataset):
	#data set class
	def __init__(self, data, target):
		self.data = torch.from_numpy(data).float()
		self.target = torch.from_numpy(target).long()

	def __getitem__(self, index):
		x = self.data[index]
		y = self.target[index]

		return x, y

	def __len__(self):
		return len(self.data)

class MLP(torch.nn.Module):
	#multi-layer perceptron class
	def __init__(self, num_fc_layers, num_fc_units, dropout_rate, n_features, n_types):
		super().__init__()
		#fc_layers = number of layers that the MLP will have
		#fc_units = number of units in each of the middle layers
		self.layers = nn.ModuleList() #empty module list as of rn
		self.layers.append(nn.Linear(n_features, num_fc_units)) #(in features, out_features)
		for i in range(num_fc_layers):
			self.layers.append(nn.Linear(num_fc_units, num_fc_units)) #linear unit
			self.layers.append(nn.ReLU(True)) #reLu activation
			self.layers.append(nn.Dropout(p=dropout_rate)) #applies dropout

		self.layers.append(nn.Linear(num_fc_units, n_types)) #in features, out_features (hard coded)
	#functions which run the model
	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.layers[i](x)

		return x

	def feature_list(self, x):
		out_list = []
		for i in range(len(self.layers)):
			x = self.layers[i](x)
			out_list.append(x)
		return out_list

	def intermediate_forward(self, x, layer_index):
		for i in range(layer_index):
			x = self.layers[i](x)
		return x

def process_data_all(test_size, n_splits, label, inputDir, outputDir):
	'''
	#Same as process_data_orig in split script, but loads pre-saved versions of training and testing set given the label
	requires path to new tables outputted from split_data
	input:
		n_splits = integer, number of ensembles to create
		label = string input from .sh file representing specific label provided for each run ('' = normal, 'bal' = balanced, can define more)
	output:
		x_train_folds, y_train_folds  = list of np.arrays representing each ensemble training set for feature tables (x_train_folds) and labels (y_train_folds)
		x_val_folds, y_val_folds  = list of np.arrays representing each ensemble training set for feature tables (x_val_folds) and labels (y_val_folds)
		x_test_folds, y_test_folds  = list of np.arrays representing each ensemble validation set for feature tables (x_test_folds) and labels (y_test_folds)
		n_features = number of distinct features
		n_types = number of distinct types
	'''
	data_train = pd.read_csv(inputDir + 'ft_train' + label + '.csv', sep = ',', squeeze = False, index_col = 0)
	labels_train = pd.read_csv(inputDir + 'labels_train' + label + '.csv', sep = ',', squeeze = True, index_col = 0)
	data_test = pd.read_csv(inputDir + 'ft_test' + label + '.csv', sep = ',', squeeze = False, index_col = 0)
	labels_test = pd.read_csv(inputDir + 'labels_test' + label + '.csv', sep = ',', squeeze = True, index_col = 0)
	n_features = len(data_test.columns)
	n_types = len(set(labels_test))
	print('n_features = ', n_features)
	print('n_types = ', n_types)
	sss = StratifiedShuffleSplit(n_splits=n_splits, random_state=0)
	sss.get_n_splits(data_train, labels_train)	#makes n_splits folds of the data, for cross validation
	encoder = LabelEncoder()
	x_train_folds, x_val_folds = [], []
	y_train_folds, y_val_folds= [], []
	for train_index, val_index in sss.split(data_train, labels_train):
		x_train, x_val = np.array(data_train.iloc[train_index]), np.array(data_train.iloc[val_index])
		y_train, y_val = np.array(labels_train.iloc[train_index]), np.array(labels_train.iloc[val_index])
		y_train = encoder.fit_transform(y_train)
		y_val = encoder.fit_transform(y_val)
		x_train_folds.append(x_train)
		x_val_folds.append(x_val)
		y_train_folds.append(y_train)
		y_val_folds.append(y_val)
	x_test = np.array(data_test)
	y_test = np.array(encoder.fit_transform(labels_test))
	return x_train_folds, x_val_folds, y_train_folds, y_val_folds, x_test, y_test, n_features, n_types

def create_loader(inputs, targets, batch_size=32):
	#loaders provide batches of X and y for evalution
	dataset = MyDataset(inputs, targets)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=2,
		pin_memory=True
		)
	return loader

def create_unshuffled_loader(inputs, targets, batch_size=32):
	#no shuffling of the loaded datasets (for model comparisons)
	dataset = MyDataset(inputs, targets)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=2,
		pin_memory=True
		)
	return loader

def evaluate_accuracy_micro(model, data_loader):
	#evaluates micro-accuracy of the model on the loaded data
	model.eval()
	correct=0
	with torch.no_grad():
		for x, y in data_loader:
			x, y = x.to(device), y.to(device)
			output = model(x)
			pred = output.max(1, keepdim=True)[1]
			correct+= pred.eq(y.view_as(pred)).sum().item()
	accuracy = correct / len(data_loader.sampler)
	return(accuracy)

#defining hyperparamter dimensions for gp_minimize
dim_learning_rate = Real(low=1e-5, high=2e-3, prior='log-uniform', name='learning_rate')
dim_weight_decay = Real(low=1e-5, high = 2e-3, prior = 'log-uniform', name='weight_decay')
dim_dropout = Real(low=1e-6, high=0.5, prior = 'log-uniform', name = 'dropout_rate')
dim_num_dense_layers = Integer(low=0, high = 3, name='num_fc_layers')
dim_num_dense_nodes = Integer(low=5, high=2048, name='num_fc_units')
dimensions= [dim_learning_rate, dim_weight_decay, dim_dropout, dim_num_dense_layers, dim_num_dense_nodes]
default_paramaters = [1e-4, 1e-3, 1e-6, 0, 100] #start values
@use_named_args(dimensions=dimensions) #accesses the list of hyper params we want to optimize
def fitness(learning_rate, weight_decay, dropout_rate, num_fc_layers, num_fc_units):
	#evaluate model given hyperparameters and saves model if the performance is better than the global best accuracy
	global best_accuracy
	global n_features
	global n_types
	#best_accuracy = 0.0
	#print()
	print(datetime.datetime.now())
	print("model fitting starts...")
	print() #print tested hyperparameters
	print('learning rate: ',learning_rate)
	print('weight_decay: ', weight_decay)
	print('dropout_rate:', dropout_rate)
	print('num_fc_layers: ', num_fc_layers + 1)
	print('num_fc_units: ', num_fc_units)
	print()
	#load model
	model = MLP(num_fc_layers, num_fc_units, dropout_rate, n_features, n_types).to(device)
	print('model built...')
	print()
	criterion = torch.nn.CrossEntropyLoss() #Log Loss function
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	#train model
	for epoch in range(200):
		loss = 0
		model.train()
		print("epoch = ", epoch, " starting..")
		for i, (x,y) in enumerate(train_loader):
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			output = model(x)
			loss = criterion(output, y)
			loss.backward()
			optimizer.step()
	#evaluate accuracy

	accuracy = evaluate_accuracy_micro(model, val_loader)
	print("done...")
	print('Micro Accuracy: {0:.2%}'.format(accuracy))
	print()
	if accuracy > best_accuracy:
		#if the model has better accuracy than the current best
		torch.save(model, path_best_model)
		best_accuracy = accuracy
	del model
	return -accuracy

if __name__ == "__main__":
	#set random seeds for consistency
	torch.manual_seed(1337)
	np.random.seed(42)

	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	use_cuda = torch.cuda.is_available()
	print('cuda = ', use_cuda)
	device = torch.device("cuda:0")
	device = torch.device(device)
	#same labels, test_size, n_splits as in split_data

	parser = argparse.ArgumentParser(
	    description='This program reads the list of command line arguments'
	)

	# Add Arguments

	parser.add_argument('--test_size', '-ts', type=int,
	                    help='percentage split for test data')


	parser.add_argument('--nsplits', '-ns', type=int,
	                    help='number of splits for cross validation')

	parser.add_argument('--inputDir', '-id', type=str, help='path of the input directory where output files from previous step have been saved')

	parser.add_argument('--outputDir', '-od', type=str, help='path of the output directory where results for this step will be saved')

	parser.add_argument('--splitFold', '-sf', type=int,
		                    help='index which grabs the correct ensemble fold')

	# Execute the parse_args() method
	args = parser.parse_args()

	test_size = args.test_size
	n_splits = args.nsplits
	label =''
	inputDir=args.inputDir
	outputDir=args.outputDir
	split=args.splitFold
	#print(outputDir)
	print()
	print ("split = ", split)
	#print()
	# if len(sys.argv) > 2:
	# 	label = '_' + sys.argv[1]
	# else:
	# 	label =''
	#load data
	#
	x_train_folds, x_val_folds, y_train_folds, y_val_folds, x_test, y_test, n_features, n_types= process_data_all(test_size, n_splits, label, inputDir, outputDir)
	print()
	print('data Pre-Processing done..')
	print()
	#split = int(sys.argv[-1]) #defined from the .sh file, index which grabs the correct ensemble fold
	x_train, y_train = x_train_folds[split], y_train_folds[split]
	x_val, y_val = x_val_folds[split], y_val_folds[split]
	#create train, val, test loaders
	train_loader = create_loader(x_train, y_train)
	val_loader = create_loader(x_val, y_val)
	test_loader = create_loader(x_test, y_test)
	path_best_model = outputDir + 'mskcl_MLP_best_split_' + str(split+1) + label + '.pt' #saves the best found model at this path
	best_accuracy = 0.0
	#gp_minimize finds the minimum of the fitness function by approximating it with a gaussian process, acquisition function over a gaussian prior chooses next param to evaluate
	#search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='gp_hedge', n_calls=500, x0=default_paramaters, random_state=7, n_jobs = -1)
	search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='gp_hedge', n_calls=500, x0=default_paramaters, random_state=7, n_jobs = -1)
	#save hyperparameters
	hyps = np.asarray(search_result.x)
	np.save(outputDir + 'mskcl_MLPsplit_' + str(split) + label + '.npy', hyps)
	#print accuracy of best result
	best_model = torch.load(path_best_model)
	test_micro = evaluate_accuracy_micro(best_model, test_loader)
	print('Split ', split)
	print('Best Validation accuracy', best_accuracy)
	print('Test micro accuracy', test_micro)
