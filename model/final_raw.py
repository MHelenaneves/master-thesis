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
from sklearn.metrics import f1_score, confusion_matrix
torch.cuda.empty_cache()
##------------##------------##------------##------------##------------##------------##------------##------------
from dataloader import ParkinsonsDataset
from networks import Net
import CV_raw_Optuna
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
        data = torch.swapaxes(data,1,0)
        y_prob_list = torch.empty((data.shape[0]), dtype=torch.float)
        y_hat_list = torch.empty((data.shape[0]), dtype=torch.float)
        y_model_output = torch.empty( (data.shape[0],2), dtype=torch.float)
        
        predicted=[]
        true=[]
        ## for each batch size
        for ii in range(data.shape[0]):
            #datain = data[ii,:,:,:]
            datain = data[ii,:,:]
            #datain = torch.squeeze(datain)
            Y_prob, Y_hat, A, output = model(datain)
            y_prob_list[ii]=Y_prob
            y_hat_list[ii]=Y_hat
            y_model_output[ii,:] = output;
            
        b = []
        for i in range(401):
            b.append( bag_label.detach().cpu().numpy() )
            # #print(np.shape(b)) #(401,1)
        acc = accuracy_score(b, y_hat_list.detach().cpu().numpy())
        train_acc_ += acc
        predicted.append(Y_hat.detach().cpu().numpy())
        true.append(bag_label.detach().cpu().numpy())
  
        
    
    # #print((y_model_output.cpu()).shape) #[401,2]
    # #print((bag_label.cpu()).shape)  #1
        b=np.array(b)
    # #bag_label=torch.nn.functional.one_hot(torch.Tensor(bag_label).to(torch.int64),2)
    # #print(bag_label)
        bag_label=torch.squeeze(torch.Tensor(b))
    # print(bag_label.shape) #[401,1]
        
        loss = loss_fn(y_model_output.cpu(),bag_label.cpu().type(torch.LongTensor))
        train_loss_ += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss_/len(train_loader)
    f1score = f1_score(true, predicted, zero_division=1)
    train_acc = train_acc_/len(train_loader)
    confusion_matrix_train=confusion_matrix(true, predicted)   

    return train_loss, f1score,train_acc


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
        data = torch.swapaxes(data,1,0)
        #data = torch.squeeze(data)
        y_prob_list = torch.empty((data.shape[0]), dtype=torch.float)
        y_hat_list = torch.empty((data.shape[0]), dtype=torch.float)
        y_model_output = torch.empty( (data.shape[0],2), dtype=torch.float)
        
        predicted=[]
        true=[]
        for ii in range(data.shape[0]):
            datain = data[ii,:,:]
            #print(datain.shape)
            Y_prob, Y_hat, A, output = model(datain)
            y_prob_list[ii]=Y_prob
            y_hat_list[ii]=Y_hat
            y_model_output[ii,:] = output;
        
        
        b = []
        for i in range(401):
           b.append( bag_label.detach().cpu().numpy() )
           
        acc = accuracy_score(b, y_hat_list.detach().cpu().numpy())
        test_acc_ += acc
        
        predicted.append(Y_hat.detach().cpu().numpy())
        true.append(bag_label.detach().cpu().numpy())
        
        #loss = criterion(mSigmoid(torch.unsqueeze(Y_prob,0).cpu()), torch.unsqueeze(bag_label,0).cpu())
        b=np.array(b)
        bag_label=torch.squeeze(torch.Tensor(b))
        #print(bag_label.shape)
        #loss = loss_fn(output.cpu(),bag_label.cpu().type(torch.LongTensor))
        #loss = criterion(mSigmoid(torch.unsqueeze(Y_prob,0).cpu()), torch.unsqueeze(bag_label,0).cpu())
        #loss = loss_fn(output.cpu(),bag_label.cpu().type(torch.LongTensor))
        loss = loss_fn(y_model_output.cpu() , bag_label.cpu().type(torch.LongTensor)) 
        test_loss_ += loss.item()
               
    test_acc = test_acc_ / len(test_loader)
    test_loss =test_loss_ / len(test_loader)
    f1score = f1_score(true, predicted,zero_division=1)
    confusion_matrix_val=confusion_matrix(true, predicted)
          
    return test_loss,f1score, test_acc

##------------##------------##------------##------------##------------##------------##---

def main():
    optimizer_name = 'Adam'
    #trial_value=CV_raw_Optuna.main()
    lr = 0.0001430
    #lr=trial_value
    use_cuda = torch.cuda.is_available() # CUDA for PyTorch
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print('The device for this training is:',device)
    loss_fn =  nn.CrossEntropyLoss();
    num_epochs = 500
    pd_data = ParkinsonsDataset()
    
    batch_size = 1
    #validation_split = .15
    shuffle_dataset = True
    random_seed= 42
    model = Net().to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    validation_split = .20
    # Creating data indices for training and validation splits:
    dataset_size = len(pd_data)
    indices = list(range(dataset_size-2))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset = pd_data,  batch_size = 1, sampler = train_sampler)
    test_loader = DataLoader(dataset = pd_data,  batch_size = 1, sampler = valid_sampler)
    if not os.path.exists('./finalmodelresults_raw/'):
        os.mkdir('./finalmodelresults_raw/')
    save_path = './finalmodelresults_raw/'
    history = {'train_loss': [], 'test_loss': [],'train_f1':[],'test_f1':[], 'train_acc':[], 'val_acc':[]}
    for epoch in range(num_epochs):
        #scheduler.step() # change learning rate with step
        #if (epoch+1)% 20 == 0:
        #    print(f'step {epoch+1}/{num_epochs}')
        #    print(optimizer)
    
        train_loss, f1score_train,train_acc = train_epoch(model,device,use_cuda,train_loader,loss_fn,accuracy_score,optimizer)
        test_loss, f1score_test, val_acc = validate_epoch(model,device,use_cuda,test_loader,loss_fn,accuracy_score)
    
        history['train_loss'].append(train_loss)
        history['train_f1'].append(f1score_train)
        history['test_loss'].append(test_loss)
        history['test_f1'].append(f1score_test)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        #history['confusion_matrix_train'].stack(confusion_matrix_train)
        #history['confusion_matrix_val'].stack(confusion_matrix_val)
        torch.save(model, save_path+'epoch'+str(epoch)+'.pt')
    
    save_str = save_path+'raw'+optimizer_name+'_'+str(lr)+'.pkl'
    cv_history = open(save_str, "wb")
    pickle.dump(history, cv_history)
    cv_history.close()
    
    return 0

if __name__ == "__main__":
    main()
