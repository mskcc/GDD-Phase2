import os, json, subprocess, csv, re, time, uuid, tempfile

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

from predict_single import EnsembleClassifier, process_data_single, MLP, pred_results



inputDir='/Users/shalabhs/Projects/GDD/Project_GDDV2/Data'
modelName='ensemble_classifier_update_bal.pt'
print("Hello SS")

torch.manual_seed(1337)
np.random.seed(42)

use_cuda = torch.cuda.is_available()
print(use_cuda)

if use_cuda:
    deviceParam="cuda:0"
    device = torch.device(deviceParam)
    torch.cuda.set_device(device)
else:
    deviceParam="cpu"
    device = torch.device(deviceParam)
    #torch.cuda.set_device(device)
    #map_location=device

print(deviceParam)



label = ''

print(os.path.join(inputDir, modelName))

fold_ensemble = torch.load(os.path.join(inputDir, modelName), map_location=device)

colnames = pd.read_csv('single_test.csv')

#colnames = pd.read_csv('/Users/shalabhs/Projects/GDD/Project_GDDV2/Data/DMPJson/features_20220519160652.%f_77a49988.txt', sep="\t")

# print(colnames)
# print(type(colnames))
print("hello SS")

# single_data = colnames
# single_data = single_data.drop(['SAMPLE_ID', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'SAMPLE_TYPE', 'PRIMARY_SITE', 'METASTATIC_SITE', 'Cancer_Type', 'Classification_Category'], axis=1)
#
# print(single_data)

torch.set_printoptions(edgeitems=3)
pred_data = process_data_single(colnames)
pred_data = torch.from_numpy(pred_data).float()
print(type(pred_data))
print(pred_data.numpy().shape)

pred_data = pred_data.to(device)

fold_logits = fold_ensemble(pred_data)

print(type(fold_logits))
print(len(fold_logits))

for i in range(len(fold_logits)):
    #print(type(fold_logits[i].detach().numpy()))
    print("Ensemble Fold = ", i)

    #print(fold_logits[i].detach().numpy().shape)
    #print(fold_logits[i].detach().numpy())
    #maxIndex=fold_logits[i].detach().numpy().argmax()
    #maxVal=fold_logits[i].detach().numpy()[12]
    print(fold_logits[i].detach().numpy())

    print("\n")
#print(fold_logits)

preds, probs = pred_results(fold_logits, label)


print(preds, probs)

res1=np.concatenate([preds, probs], axis=None, dtype=object)
# res = pd.DataFrame([preds,probs]).T
res = pd.DataFrame(res1).T
res.columns = ['pred1','pred2','pred3','prob1','prob2', 'prob3' ]
res.to_csv('single_res.csv',index=False)
