# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:37:26 2022

@author: mhele
"""
import pandas as pd
import numpy as np
from PreProcessing import get_all_data, check_recordings_data,filtering_emg_alt
import scipy.io
from sklearn.preprocessing import StandardScaler
    

def get_data_matrix():
    data=[]
    for ID in range(1,17): 
        obs_windows_all=[]
        print(ID)
        
        if ID==8:
            ID=ID+1
        else:
            obs_windows_all=[]
            obs_windows_all=sub_windows_EMG(ID)
        
            if np.size(data,0)==0:
                data = obs_windows_all
            else:
                data=np.dstack((data,obs_windows_all))
        #print(np.size(data,2))
        
    print("done")
    return data
            


def sub_windows_EMG(ID): #create a matrix of windows of EMG data(each row is a window, and each window is 6000 samples (3s))

    window_labels=get_window_labels(ID, 1)
    max_restwindows= get_max_restwindows()
    emg_filtered= get_EMG_sub(ID)
    
    obs_windows=[]
    window_length=3*2000
    j=0
    
    for i in range(len(window_labels)):
        k = j + window_length
    
        if window_labels[i]==1:
            k = j + window_length
            window_rest=emg_filtered[j:k] 
            
            if len(obs_windows)==0:
                obs_windows = np.append(obs_windows, window_rest)
            else:
                obs_windows = np.vstack((obs_windows, window_rest))
                #print(np.size(obs_windows,1))
        j = k
    
    if len(obs_windows) < max_restwindows:
        addition= np.tile(obs_windows[0,:],((int(max_restwindows-len(obs_windows))),1))
        obs_windows_all= np.vstack((obs_windows,addition))
    if len(obs_windows) == max_restwindows:
        obs_windows_all= obs_windows

    
    #print("done")
    return obs_windows_all

def sub_windows_gyro(ID): #create a matrix of windows of EMG data (each row is a window, and each window is 6000 samples) (3s))

    window_labels=get_window_labels(ID, 1)
    max_restwindows= get_max_restwindows()
    gyro_x, gyro_y, gyro_z= get_gyro_sub(ID)
    
    gyro_vector_magnitude = np.sqrt(((gyro_x)**2 + (gyro_y)**2 + (gyro_z**2)))

    
    obs_windows=[]
    window_length=3*2000
    j=0
    
    for i in range(len(window_labels)):
        k = j + window_length
    
        if window_labels[i]==1:
            k = j + window_length
            window_rest=gyro_vector_magnitude[j:k] 
            
            if len(obs_windows)==0:
                obs_windows = np.append(obs_windows, window_rest)
            else:
                obs_windows = np.vstack((obs_windows, window_rest))
                #print(np.size(obs_windows,1))
        j = k
    
    if len(obs_windows) < max_restwindows:
        addition= np.tile(obs_windows[0,:],((int(max_restwindows-len(obs_windows))),1))
        obs_windows_all= np.vstack((obs_windows,addition))
    if len(obs_windows) == max_restwindows:
        obs_windows_all= obs_windows
    
    #print("done")
    return obs_windows_all

def get_EMG_sub(ID):
    
    file_data= get_all_data()[(ID+15)]
    EMG= check_recordings_data(file_data)[0]
    emg_filtered=filtering_emg_alt(EMG)[2] #correct this so we normalize per window
    return EMG, emg_filtered

def get_gyro_sub(ID):
    
    file_data= get_all_data()[(ID+15)]
    Gyro_X,Gyro_Y, Gyro_Z= check_recordings_data(file_data)[4], check_recordings_data(file_data)[5],check_recordings_data(file_data)[6]
    return Gyro_X,Gyro_Y, Gyro_Z



def get_window_labels(ID, param):
    all_files_output = {}
    itr = iter(range(1,10))
    if ID in itr:
        if ID==8:
            ID=ID+1
        else: 
            s = 'C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/Code/Output Hand classifier/window_labels_ID0{}_T0.001'.format(ID)
            all_files_output = pd.read_csv(s +'.csv')

    else:
        s = 'C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/Code/Output Hand classifier/window_labels_ID{}_T0.001'.format(ID) 
        all_files_output = pd.read_csv(s +'.csv')
    
    if param==0:
        return all_files_output

    if param==1:
        
        windowtemp_labels = all_files_output.values
        window_labels = []
        
        #print(windowtemp_labels[0])
        for i in range(len(windowtemp_labels)):
            #window_labels+=[elem[0]]
            window_labels=np.append(window_labels,windowtemp_labels[i][0])
            
        return window_labels


def get_max_restwindows(): 
    nwindows_rest_all=[]
    #itr = iter(range(1,17))

    for ID in range(1,17):
        #print(ID)
        
        if ID==8:
            ID=ID+1
        else:
            #print(ID)
            window_labels=get_window_labels(ID, 1)
        
            nwindows_rest=[]
            for j in range(len(window_labels)):
                if int(window_labels[j])==0:
                    nwindows_rest= np.append(nwindows_rest,window_labels[j])
                windows_rest=len(nwindows_rest)
                #print(windows_rest)
            nwindows_rest_all=np.append(nwindows_rest_all,windows_rest)
        
            max_restwindows= np.max(nwindows_rest_all)
    
    return  max_restwindows 

#%% Check StandardScaler    

scaler = StandardScaler()
standardized = scaler.fit_transform(EMG)
scaler.fit(EMG)

#%% Save data matrix

# Some test data
data=get_data_matrix()

# Specify the filename of the .mat file
matfile = 'data_matrix.mat'

# Write the array to the mat file. For this to work, the array must be the value
# corresponding to a key name of your choice in a dictionary
scipy.io.savemat(matfile, mdict={'out': data}, oned_as='row')

#%% Load data matrix

# Now load in the data from the .mat that was just saved
matdata = scipy.io.loadmat(matfile)

# And just to check if the data is the same:
assert np.all(data == matdata['out'])      
data = matdata['out']
        