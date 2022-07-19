import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Dropout
#from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
import pickle
from sklearn.model_selection import KFold
import optuna
from optuna.trial import TrialState
from torchsummary import summary


# Model that takes the raw filtered data of 12 channels
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.L = 16
        self.K = 1
        self.M = 64
        ## feature extraction
        self.featureExtract = nn.Sequential(nn.Conv1d(in_channels = 1 , out_channels = 32, kernel_size=8, padding =1,stride = 1),
                                            #nn.BatchNorm1d(5995),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.MaxPool1d(2),
                                            #nn.AvgPool1d(2),
                                            Dropout(p=0.3, inplace=False),

                                            nn.Conv1d(in_channels = 32,out_channels = 32, kernel_size=8,padding =1, stride = 1),
                                            #nn.BatchNorm1d(2992),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.MaxPool1d(2),
                                            #nn.AvgPool1d(2),
                                            Dropout(p=0.3, inplace=False),

                                            nn.Conv1d(in_channels = 32, out_channels = 16, kernel_size=16, padding =1, stride = 1),
                                            #nn.BatchNorm1d(1483),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.MaxPool1d(2),
                                            #nn.AvgPool1d(2),
                                            Dropout(p=0.3, inplace=False),

                                            nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size=16, padding =1, stride = 1),
                                            #nn.BatchNorm1d(728),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.MaxPool1d(2),
                                            #nn.AvgPool1d(2),
                                            Dropout(p=0.3, inplace=False),

                                            nn.Flatten(),
                                            nn.Linear(224, self.M))

        self.attention = nn.Sequential(nn.Linear(self.M, self.L),
                                       nn.Tanh(),
                                       nn.Linear(self.L, self.K))

        self.classify = nn.Sequential(nn.Linear(self.M,32),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      Dropout(p = 0.5, inplace = False),
                                      nn.Linear(32,16),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      Dropout(p = 0.6, inplace = False),
                                      nn.Linear(16,2),
                                      nn.Softmax(dim = 1))

    def forward(self, x):
        ## feature extraction

        H = self.featureExtract(x)

        A = self.attention(H)
        A = torch.transpose(A,1,0)
        s = nn.Softmax(dim=1)
        A = s(A)

        z = torch.mm(A,H)

        output = self.classify(z)
        Y_prob , Y_hat = torch.max(output,dim=1)

        return Y_prob, Y_hat, A, output

model=Net()
#batch_size=1
#summary(model, input_size=(batch_size,1,401,6000))

def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat



