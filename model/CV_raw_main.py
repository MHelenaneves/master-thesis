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
torch.cuda.empty_cache()
##------------##------------##------------##------------##------------##------------##------------##------------
from dataloader import ParkinsonsDataset
from networks import Net
##------------##------------##------------##------------##------------##------------##------------##------------

def train_epoch(model,device,use_cuda,train_loader,loss_fn,accuracy_score,optimizer):
    # Train model:
    model.train()
    train_loss_ = 0.
    train_acc_ = 0.
    for i, (data, bag_label) in enumerate(train_loader):

        if use_cuda:
            data, bag_label = data.to(device, dtype=torch.float), bag_label.to(device, dtype=torch.float)
        else:
            data, bag_label = data.type(torch.float), bag_label.type(torch.float)

        # wrap them in Variable
        data, bag_label = Variable(data), Variable(bag_label)
        #data = torch.swapaxes(data,3,2)
        y_prob_list = torch.empty((data.shape[0]), dtype=torch.float)
        y_hat_list = torch.empty((data.shape[0]), dtype=torch.float)
        y_model_output = torch.empty( (data.shape[0],2), dtype=torch.float)

        ## for each batch size
        for ii in range(data.shape[0]):
            datain = data[ii,:,:]
            datain = torch.squeeze(datain)
            Y_prob, Y_hat, A, output = model(datain)
            y_prob_list[ii]=Y_prob
            y_hat_list[ii]=Y_hat
            y_model_output[ii,:] = output;

        acc = accuracy_score(bag_label.detach().cpu().numpy(), y_hat_list.detach().cpu().numpy())
        train_acc_ += acc
        loss = loss_fn(y_model_output.cpu(),bag_label.cpu().type(torch.LongTensor))
        train_loss_ += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss_/len(train_loader)
    train_acc = train_acc_/len(train_loader)

    return train_loss, train_acc


##------------##------------##------------##------------##------------##------------##------------##------------
def validate_epoch(model,device,use_cuda,test_loader,loss_fn,accuracy_score):
    # test model:
    model.eval()
    test_loss_ = 0.
    test_acc_ = 0.
    for i, (data, bag_label) in enumerate(test_loader):
        if use_cuda:
            data, bag_label = data.to(device, dtype=torch.float), bag_label.to(device, dtype=torch.float)
        else:
            data, bag_label = data.type(torch.float), bag_label.type(torch.float)

            # wrap them in Variable
        data, bag_label = Variable(data), Variable(bag_label)
        data = torch.swapaxes(data,3,2)
        data = torch.squeeze(data)

        Y_prob, Y_hat, A, output = model(data)
        acc = accuracy_score(bag_label.detach().cpu().numpy(), Y_hat.detach().cpu().numpy())
        test_acc_ += acc

        #loss = criterion(mSigmoid(torch.unsqueeze(Y_prob,0).cpu()), torch.unsqueeze(bag_label,0).cpu())
        loss = loss_fn(output.cpu(),bag_label.cpu().type(torch.LongTensor))
        test_loss_ += loss.item()

    test_acc =test_acc_ / len(test_loader)
    test_loss =test_loss_ / len(test_loader)

    return test_acc,test_loss

##------------##------------##------------##------------##------------##------------##---
#################################Cross Validation loop #################################
# defining the CV-fold settings:
k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

##------------##------------##------------##------------##------------##------------##---
################################# Some hyperparameters #################################


use_cuda = torch.cuda.is_available() # CUDA for PyTorch
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('The device for this training is:',device)
loss_fn =  nn.CrossEntropyLoss();
foldperf={}
num_epochs = 80
pd_data = ParkinsonsDataset()
lr = 0.0001
optimizer_name = 'Adam'
##------------##------------##------------##------------##------------##------------##---

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(pd_data)))):

    model = Net().to(device)
    #optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset = pd_data,  batch_size = 1, sampler = train_sampler)
    test_loader = DataLoader(dataset = pd_data,  batch_size = 1, sampler = valid_sampler)

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    for epoch in range(num_epochs):
        #scheduler.step() # change learning rate with step
        if (epoch+1)% 20 == 0:
            print(f'step {epoch+1}/{num_epochs}')
            print(optimizer)

        train_loss, train_acc = train_epoch(model,device,use_cuda,train_loader,loss_fn,accuracy_score,optimizer)
        test_acc,test_loss = validate_epoch(model,device,use_cuda,test_loader,loss_fn,accuracy_score)

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                 num_epochs,
                                                                                                                 train_loss,
                                                                                                                 test_loss,
                                                                                                                 train_acc,
                                                                                                                 test_acc))
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

    foldperf['fold{}'.format(fold+1)] = history

save_path = '/zhome/37/8/118154/Documents/ParamSearch/FinalOptuna/raw/'
save_str = save_path+'raw'+optimizer_name+'_'+str(lr)+'.pkl'
cv_history = open(save_str, "wb")
pickle.dump(foldperf, cv_history)
cv_history.close()
