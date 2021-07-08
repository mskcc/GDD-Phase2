import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
print('imports finished')

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

		self.layers.append(nn.Linear(num_fc_units, 41)) #(in features, out_features)
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

def process_data(): 
	#Same as process_data_orig in split script, but loads pre-saved versions of training and testing set given the label
	#returns train and test splits prior to splitting into ensemble folds
	x_train = pd.read_csv('ft_train' + label + '.csv', sep = ',', squeeze = False, index_col = 0)
	y_train = pd.read_csv('labels_train' + label + '.csv', sep = ',', squeeze = True, index_col = 0)
	x_test = pd.read_csv('ft_test' + label + '.csv', sep = ',', squeeze = False, index_col = 0)
	y_test = pd.read_csv('labels_test' + label + '.csv', sep = ',', squeeze = True, index_col = 0)
	encoder = LabelEncoder()
	y_train = encoder.fit_transform(y_train)
	y_test = encoder.fit_transform(y_test)
	return np.array(x_train), np.array(x_test), y_train, y_test

def evaluate_accuracy(model, data_loader): 
	#evaluate model and return accuracy, true Y, predicted Y, probability of Y for further analysis
	model.eval()
	correct=0
	y_true = []
	y_preds = []
	y_probs = []
	with torch.no_grad(): 
		for x, y in data_loader:
			#this will load the validation data and give the labels 
			y_true.append(y.data.numpy())
			x, y = x.to(device), y.to(device)
			output = model(x)
			_, pred = output.max(1, keepdim=True)
			soft_out = F.softmax(output, dim = 1)
			pred_probs, _ = torch.max(soft_out, 1)
			correct+= pred.eq(y.view_as(pred)).sum().item()			
			y_preds.append(pred.cpu().data.numpy())	
			y_probs.append(pred_probs.cpu().data.numpy())
	accuracy = correct / len(data_loader.sampler)
	print(accuracy)
	return(accuracy, y_true, y_preds, y_probs)


def softmax_predictive_accuracy(logits_list, y, label):
	#ensemble accuracy function which evaluates accuracy and saves stats for further analysis
    probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
    fold_preds = [torch.max(probs_list[i], 1)[1].cpu().data.numpy() for i in range(len(probs_list))]
    n_dif_preds = []
    for i in range(len(fold_preds[0])):
    	pred_types = [fold_preds[j][i] for j in range(len(fold_preds))]
    	n_dif_preds.append(len(set(pred_types)))
    np.savetxt('ens_n_preds.csv', n_dif_preds, delimiter = ',')
    probs_tensor = torch.stack(probs_list, dim = 2)
    probs = torch.mean(probs_tensor, dim=2)
    pred_probs, pred_class = torch.max(probs, 1)
    probs = probs.cpu().data.numpy()
    pred_probs = pred_probs.cpu().data.numpy()
    preds = pred_class.cpu().data.numpy()
    np.savetxt('ens_allprobs' + label + '.csv', probs, delimiter = ',')
    np.savetxt('ens_probs' + label + '.csv',pred_probs, delimiter = ',')
    np.savetxt('ens_preds' + label + '.csv',preds, delimiter = ',')
    correct = (pred_class == y)
    np.savetxt('ens_true' + label + '.csv', y.cpu().data.numpy(), delimiter = ',')
    pred_acc = correct.float().mean()
    return pred_acc

if __name__ == "__main__":
	#set seeds and load
	torch.manual_seed(1337)
	np.random.seed(42)
	#use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
	n_splits = 10
	if len(sys.argv) > 1:
		label = '_' + sys.argv[-1]
	else:
		label = ''
	#process and load data
	x_train, x_test, y_train, y_test = process_data()
	train_loader = create_loader(x_train, y_train)
	test_loader = create_loader(x_test, y_test)
	fold_model_list = [] #list of each models
	fold_acc_list = [] #list of respective accuracies
	#define overall true, pred, prob
	all_y_true = [] 
	all_y_pred = []
	all_y_probs = []
	#over each fold:
	for i in range(1, n_splits+1): 
		#load model
		path_best_model = './mskcl_MLP_best_split_' + str(i) + label + '.pt' #saves the best found model at this path
		model = torch.load(path_best_model)
		print('i: ', i)
		#evaluate accuracy
		evaluate_accuracy(model, test_loader)
		#append to overall list
		fold_model_list.append(model)
	#create full ensemble, save
	fold_ensemble = EnsembleClassifier(fold_model_list) #create full ensemble, save
	torch.save(fold_ensemble, 'ensemble_classifier_update' + label + '.pt')
	#load and report overall accuracy
	x_test = torch.from_numpy(x_test).float()
	y_test = torch.from_numpy(y_test).long()
	x_test, y_test = x_test.to(device), y_test.to(device)
	fold_logits = fold_ensemble(x_test)
	fold_acc = softmax_predictive_accuracy(fold_logits, y_test, label)
	fold_acc = np.float(fold_acc.cpu().numpy())
	print(fold_acc)
	
	


