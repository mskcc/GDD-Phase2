import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import scipy

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys

from train_gdd_nn import MLP, MyDataset, create_loader, create_unshuffled_loader

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
	return(accuracy)


def softmax_predictive_accuracy(logits_list, y, label):
    #ensemble accuracy function which evaluates accuracy and saves stats for further analysis
    probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
    for i in range(len(logits_list)):
        logits = logits_list[i]
        log_form = logits.cpu().data.numpy()
    fold_preds = [torch.max(probs_list[i], 1)[1].cpu().data.numpy() for i in range(len(probs_list))]

    #number of different predictions across folds
    n_dif_preds = []
    for i in range(len(fold_preds[0])):
        pred_types = [fold_preds[j][i] for j in range(len(fold_preds))]
        n_dif_preds.append(len(set(pred_types)))

    #probs, preds
    probs_tensor = torch.stack(probs_list, dim = 2)
    probs = torch.mean(probs_tensor, dim=2)
    pred_probs, pred_class = torch.max(probs, 1)
    probs = probs.cpu().data.numpy()
    pred_probs = pred_probs.cpu().data.numpy()
    preds = pred_class.cpu().data.numpy()
    np.savetxt('ensemble_allprobs' + label + '.csv', probs, delimiter = ',')
    correct = (pred_class == y)
    pred_acc = correct.float().mean()

    #new stats
    allpreds = pd.DataFrame(probs)
    entropy = [scipy.stats.entropy(allpreds.iloc[i]) for i in allpreds.index]
    entropy_list = []
    for tensor_temp in probs_list:
        tensor = tensor_temp.cpu().data.numpy()
        tens_ent = [scipy.stats.entropy(i) for i in tensor]
        entropy_list.append(tens_ent)
    entropy_list = pd.DataFrame(entropy_list)
    alleo_entropy = entropy_list.mean(axis=0)

    #save data
    org_data = pd.DataFrame(y.cpu().data.numpy())
    org_data.columns = ['true']
    org_data = org_data.assign(pred = preds)
    org_data = org_data.assign(probs = pred_probs)
    org_data = org_data.assign(n_preds = n_dif_preds)
    org_data = org_data.assign(entropy = entropy)
    org_data = org_data.assign(alleotoric_entropy = alleo_entropy)
    org_data = org_data.assign(info_gain = org_data.entropy - org_data.alleotoric_entropy)
    org_data.to_csv('ensemble_results' + label + '.csv', index=False)
    return pred_acc

if __name__ == "__main__":
	#set seeds and load
	torch.manual_seed(3407)
	np.random.seed(0)
	use_cuda = torch.cuda.is_available()
	print('cuda = ', use_cuda)
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
	n_splits = 10
	
	if len(sys.argv) > 1:
		label = '_' + sys.argv[1]
		print(label)
	else:
		label = ''

	#process and load data
	x_train, x_test, y_train, y_test = process_data()
	train_loader = create_unshuffled_loader(x_train, y_train)
	test_loader = create_unshuffled_loader(x_test, y_test)
	x_test = torch.from_numpy(x_test).float()
	y_test = torch.from_numpy(y_test).long()
	x_test, y_test = x_test.to(device), y_test.to(device)
	
	#aggreagate ensemble
	fold_model_list = [] #list of each models
	for i in range(1, n_splits+1): 
		#load model
		path_best_model = './fold_'+ str(i) +  label + '.pt' 
		model = torch.load(path_best_model)
		#evaluate accuracy
		acc = evaluate_accuracy(model, test_loader)
		#append to overall list
		fold_model_list.append(model)

	#create full ensemble, save
	fold_ensemble = EnsembleClassifier(fold_model_list)
	torch.save(fold_ensemble, 'ensemble' + label + '.pt')

	#report accuracy
	fold_logits = fold_ensemble(x_test)
	fold_acc = softmax_predictive_accuracy(fold_logits, y_test,label)
	fold_acc = np.float(fold_acc.cpu().numpy())
	print('final accuracy=', fold_acc)
