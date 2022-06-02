# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 00:04:17 2022

@author: mhele
"""
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


import scipy
from torch.utils.data import Dataset, DataLoader



class ModelData():
    data = None
    batch_size = 1
    def __init__(self):
        matfile = 'data_matrix.mat'
        #data=get_data_matrix()
        # Now load in the data from the .mat that was just saved
        matdata = scipy.io.loadmat(matfile)
        # And just to check if the data is the same:
    
        #assert np.all(data == matdata['out'])      
        self.data = matdata['out']
        

    def createTrainSampler(self):

    def createValidSampler(self):
        

    def createDataset(self):
        tensor = torch.from_numpy(self.data)
        torch.utils.data.DataLoader(tensor, batch_size = batch_size, sampler)
        
