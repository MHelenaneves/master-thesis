# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:19:27 2022

@author: mhele
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def ConvertCV(foldperf):
    testl_f = np.zeros((5,250)) #80 epochs
    tl_f= np.zeros((5,250))
    testa_f= np.zeros((5,250))
    ta_f= np.zeros((5,250))
    k=5
    for f in range(1,k+1):
        tl_f[f-1,:] = foldperf['fold{}'.format(f)]['train_loss']
        ta_f[f-1,:] = foldperf['fold{}'.format(f)]['train_acc']
        testl_f[f-1,:]=foldperf['fold{}'.format(f)]['test_loss']
        testa_f[f-1,:]=foldperf['fold{}'.format(f)]['test_acc']
    cv_test_l = np.mean(testl_f,axis=0) 
    cv_test_ac = np.mean(testa_f,axis=0) 
    cv_train_l = np.mean(tl_f,axis=0) 
    cv_train_ac = np.mean(ta_f,axis=0) 
    return cv_train_l, cv_test_l, cv_train_ac, cv_test_ac

#path = "C:\\Users\\mhele\\OneDrive\\Ambiente de Trabalho\\DTU\\2nd year\\Thesis\\master-thesis\\Optuna_raw_n8"
path="C:\\tmp_On3_E250_samedataeveryfold"
LearningRates = glob.glob(path + "\*.pkl")

for i in range(len(LearningRates)):
    #fig=plt.figure(figsize=(18,6))
    plt.figure()
    legends = []
    #plt.subplot(1,4,1)
    p = 0
    #for path in LearningRates: #for each file we check the 5 folds, and in each fold  we check train and test loss, and train and test accuracy
    #print("here"+ path)
    a_file=open(LearningRates[i], "rb")
    foldperf=pickle.load(a_file)
    cv_train_l, cv_test_l, _, _ = ConvertCV(foldperf)
    plt.plot(cv_train_l, color="#5758BB", label="Train Loss")
    plt.plot(cv_test_l, color="#ED4C67", label="Validation Loss")
    #legends.append(p)
        #p+=1
    plt.legend() 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.title("CV Train Loss")
    
       
    # plt.subplot(1,4,2)
    # mins =[]
    # #for path in LearningRates:
    # a_file = open(LearningRates[i], "rb")
    # foldperf= pickle.load(a_file)
    # cv_train_l, cv_test_l, _, _ = ConvertCV(foldperf) 
    # plt.plot(cv_test_l,color="#5758BB")
    # mins.append(np.min(cv_test_l))
    # plt.legend(legends)
    # plt.title("CV Test Loss")
    
    # plt.subplot(1,4,3)
    # mins =[]
    # #for path in LearningRates:
    # a_file = open(LearningRates[i], "rb")
    # foldperf= pickle.load(a_file)
    # _, _, cv_train_ac, cv_test_ac= ConvertCV(foldperf) 
    # plt.plot(cv_train_ac)
    # plt.legend(legends)
    # plt.title("CV Train Accuracy")
    
    # plt.subplot(1,4,4)
    # mins =[]
    # #for path in LearningRates:
    # a_file = open(LearningRates[i], "rb")
    # foldperf= pickle.load(a_file)
    # _, _, cv_train_ac, cv_test_ac= ConvertCV(foldperf) 
    # plt.plot(cv_test_ac)
    # plt.legend(legends)
    # plt.title("CV Test Accuracy")
    
#%%
cv_train_loss=[]
cv_test_loss=[]
cv_train_acc=[]
cv_test_acc=[]
for i in range(len(LearningRates)):
    a_file=open(LearningRates[i], "rb")
    foldperf=pickle.load(a_file)
    cv_train_l, cv_test_l, cv_train_ac, cv_test_ac = ConvertCV(foldperf)
    if len(cv_train_loss)==0:
        cv_train_loss=np.append(cv_train_loss,cv_train_l)
        cv_test_loss=np.append(cv_test_loss,cv_test_l)
        cv_train_acc=np.append(cv_train_acc,cv_train_ac)
        cv_test_acc=np.append(cv_test_acc,cv_test_ac)
    else:
       cv_train_loss=np.vstack((cv_train_loss,cv_train_l))
       cv_test_loss=np.vstack((cv_test_loss,cv_test_l))
       cv_train_acc=np.vstack((cv_train_acc,cv_train_ac))
       cv_test_acc=np.vstack((cv_test_acc,cv_test_ac))


#%%
plt.figure()
x=range(1,81)
y=cv_test_loss[0]
y1=cv_test_loss[1] 
y2=cv_test_loss[2] 
y3=cv_test_loss[3]
y4=cv_test_loss[4]
y5=cv_test_loss[5]
plt.ylabel("Validation Loss")
plt.xlabel("Epochs")
plt.plot(x,y, label="trial1")
plt.plot(x,y1,label="trial2")
plt.plot(x,y2,label="trial3")
plt.plot(x,y3,label="trial4")
plt.plot(x,y4,label="trial5")
plt.plot(x,y5,label="trial6")
plt.legend()
plt.show()
#%%

dirs=os.listdir("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Optuna_raw_n8")
lr_array=[]
for i in range(len(dirs)):
    lr=dirs[i][8:30]
    lr_array=np.append(lr_array,lr)
    
plt.figure()
y=lr_array
x=range(1,7)
plt.scatter(x,y)
plt.ylabel("Learning Rate")
plt.xlabel("Trials")
    
    
