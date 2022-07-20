import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from sklearn.metrics import accuracy_score
import torch.optim as optim
import glob
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
import os
from os import listdir
from os.path import join, isfile
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from datetime import date, datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from dataloader import ParkinsonsDataset
from networks import Net

pd_data = ParkinsonsDataset()

dataset_size = 32
indices = list(range(2,32))
split = int(np.floor(0.2 * dataset_size))

#train_indices, val_indices = indices[split:], indices[:split]

test_indices=[1]

test_sampler=SubsetRandomSampler(test_indices)

test_loader = DataLoader(dataset = pd_data,  batch_size = 1, sampler = test_sampler)

dataiter = iter(test_loader)

data_ = dataiter.next()

data, bag_label = data_

data, bag_label = Variable(data), Variable(bag_label)

data=torch.swapaxes(data,0,1)

#data = torch.squeeze(data)
data=np.transpose(data)

data, bag_label = data.type(torch.float), bag_label.type(torch.float)   

model = Net()
#check_model = model(data)
Y_prob2, Y_hat2, A2, output2= model(data)