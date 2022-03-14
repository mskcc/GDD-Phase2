import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
import scipy.stats

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

def create_loader(inputs, targets, batch_size=32): 
	#provide batches of X and y for evalution
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
	#provide batches of X and y for evalution, unshuffled (for comparisons)
	dataset = MyDataset(inputs, targets)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=2,
		pin_memory=True
		)
	return loader 


class MLP(torch.nn.Module):
	#multi-layer perceptron class
	def __init__(self, num_fc_layers, num_fc_units, dropout_rate):
		super().__init__()
		#fc_layers = number of layers that the MLP will have
		#fc_units = number of units in each of the middle layers
		self.layers = nn.ModuleList() #empty module list as of rn
		self.layers.append(nn.Linear(5618, num_fc_units)) #(in features, out_features)
		for i in range(num_fc_layers):
			self.layers.append(nn.Linear(num_fc_units, num_fc_units))
			self.layers.append(nn.ReLU(True))
			self.layers.append(nn.Dropout(p=dropout_rate))

		self.layers.append(nn.Linear(num_fc_units, 42)) #(in features, out_features)
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

class EnsembleClassifier(nn.Module):
	#Ensemble Class
	def __init__(self, model_list): 
		super(EnsembleClassifier, self).__init__()
		self.model_list = model_list

	def forward(self, x): 
		logit_list = [] 
		for model in self.model_list: 
			model.eval()
			logits = model(x)
			logit_list.append(logits)
		return logit_list 

def process_data_single(colnames): 
	### process_data_all will create train, test, validation folds for all classes with min_samples number of samples
	single_data = pd.read_csv('single_test.csv') 
	print(single_data)
	single_data = single_data.drop(['SAMPLE_ID', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'SAMPLE_TYPE', 'PRIMARY_SITE', 'METASTATIC_SITE', 'Cancer_Type', 'Classification_Category'], axis=1)

	single_data = single_data[[i for i in colnames if i in single_data.columns]]

	print(single_data)
	return np.array(single_data)

def pred_results(logits_list, label):
	#similar to softmax_predictive_accuracy function
	probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
	fold_preds = [torch.max(probs_list[i], 1)[1].cpu().data.numpy() for i in range(len(probs_list))]
	probs_tensor = torch.stack(probs_list, dim = 2)
	probs = torch.mean(probs_tensor, dim=2)
	pred_probs, pred_class = torch.max(probs, 1)
	probs = probs.cpu().data.numpy()
	pred_probs = pred_probs.cpu().data.numpy()
	preds = pred_class.cpu().data.numpy()
	return preds, pred_probs


if __name__ == "__main__":
	torch.manual_seed(1337)
	np.random.seed(42)
	print('single prediction')
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
	label = ''
	fold_ensemble = torch.load('ensemble_classifier_update' + label + '.pt')
	colnames = pd.read_csv('col_names.csv')
	pred_data = process_data_single(colnames)
	pred_data = torch.from_numpy(pred_data).float()
	pred_data = pred_data.to(device)
	fold_logits = fold_ensemble(pred_data)
	preds, probs = pred_results(fold_logits, label) #should be zero or close to it
	res = pd.DataFrame([preds,probs]).T
	res.columns = ['pred', 'prob']
	res.to_csv('single_res.csv')
