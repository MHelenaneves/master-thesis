# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:35:28 2022

@author: mhele
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

check_results5=glob.glob("C:/tmp_run5/rawAdam_0.0001542.pkl")
data5=pickle.load( open("C:/tmp_run5/rawAdam_0.0001542.pkl", 'rb'))

check_results9=glob.glob("C:/tmp_run9/rawAdam_0.0001542.pkl")
data9=pickle.load( open("C:/tmp_run9/rawAdam_0.0001542.pkl", 'rb'))

#%% Validation and accuracy 

plt.figure()
plt.plot(data9["train_loss"],color="#5758BB", label="Train Loss")
plt.plot(data9["test_loss"],color="#ED4C67", label="Validation Loss")
plt.legend(loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.figure()
plt.plot(data9["train_acc"],color="#5758BB", label="Train Accuracy")
plt.plot(data9["val_acc"],color="#ED4C67", label="Validation Accuracy")
plt.legend(loc="upper left")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

#%% Different runs in the same plot
plt.figure()
plt.plot(data4["test_loss"],color="#5758BB", label="Run 4")
plt.plot(data5["test_loss"],color="#ED4C67", label="Run 5")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

#%% Separate plots
plt.figure()
plt.plot(data4["test_loss"],color="#5758BB", label="Run 4")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

plt.figure()
plt.plot(data5["test_loss"],color="#ED4C67", label="Run 5")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

#%%

train_acc_5=np.mean(data5["train_acc"])
val_acc_5=np.mean(data5["val_acc"])