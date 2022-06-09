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
from spectrum import arburg, pburg, data_two_freqs,arma2psd
import scipy as sp
from scipy import signal
import os

def obs_leftovers_HC(leftovers_all_HC,min_restwindows):
    path_HC=os.path("./Observations/HC")
    
    
    n_times_HC=np.floor(len(leftovers_all_HC)/min_restwindows) #it will be 1
    obs=leftovers_all_HC[0:(min_restwindows+1)]
    np.save(os.path.join(path_HC,"observation10"),obs)
    
    
    
    
def obs_leftovers_PD(leftovers_all_PD,min_restwindows):
    path_PD=os.path("./Observations/PD")
    n_times_PD=np.floor(len(leftovers_all_PD)/min_restwindows) #it will be 4
    j = 11
    for i in range(n_times_PD):
        obs = leftovers_all_PD[min_restwindows*i:(min_restwindows*(i+1))+1]
        np.save(os.path.join(path_PD,"observation%02d" % j ),obs)
        j += 1
    

#%%
def create_observations():
    parent_dir=create_directories()
    leftovers_all_HC=create_observations_HC(parent_dir)
    leftovers_all_PD=create_observations_PD(parent_dir)
    return leftovers_all_HC,leftovers_all_PD

def create_observations_HC(parent_dir):
    nwindows_rest_all, min_restwindows=get_min_restwindows()
    min_restwindows=int(min_restwindows)
    #max_restwindows=np.max(nwindows_rest_all)
    #max_nobsmatrix=np.floor(max_restwindows, min_restwindows) #2 for my data
    #parent_dir=create_directories()
    path_HC=os.path.join(parent_dir, "HC")
    os.mkdir(path_HC)
    j = 0
    leftovers_all_HC=[]
    
    for ID in range(10,15):
        obs_windows=sub_windows_EMG(ID) #all the rest windows of a sub
        if len(obs_windows)== min_restwindows:
            ID=ID+1
        else: 
            j+=1
            n_times_sub=np.floor(nwindows_rest_all[(ID)]/min_restwindows)
            if n_times_sub == 1:
                
                obs=obs_windows[0:(min_restwindows+1)]
                np.save(os.path.join(path_HC,"observation%02d" % j),obs)
                leftovers= obs_windows[(min_restwindows+1):-1]
                if len(leftovers_all_HC)==0:
                    leftovers_all_HC=leftovers
                else:
                    leftovers_all_HC=np.vstack((leftovers_all_HC, leftovers))
                    
            elif n_times_sub == 2:
                obs=obs_windows[0:(min_restwindows+1)]
                np.save(os.path.join(path_HC,"observation%02d" % j),obs)
                j+=1
                obs1= obs_windows[(min_restwindows+1):((2*min_restwindows)+1)]
                np.save(os.path.join(path_HC,"observation%02d" % j),obs1)
    
                leftovers=obs_windows[((2*min_restwindows)+1):-1]
                if len(leftovers_all_HC)==0:
                    leftovers_all_HC=leftovers
                else:
                    leftovers_all_HC=np.vstack((leftovers_all_HC, leftovers))
    
    return leftovers_all_HC



def create_observations_PD(parent_dir): 
    nwindows_rest_all, min_restwindows=get_min_restwindows()
    min_restwindows=int(min_restwindows)
    #max_restwindows=np.max(nwindows_rest_all)
    #max_nobsmatrix=np.floor(max_restwindows, min_restwindows) #2 for my data
    #parent_dir=create_directories()
    path_PD=os.path.join(parent_dir, "PD")
    os.mkdir(path_PD)
    j = 0
    leftovers_all_PD=[]
    for ID in range(1,12): 
        if ID==8:
            ID=ID+1
        elif ID>8:
            obs_windows=sub_windows_EMG(ID) #all the rest windows of a sub
            if len(obs_windows)== min_restwindows:
                ID=ID+1
            else: 
                j+=1
                n_times_sub=np.floor(nwindows_rest_all[(ID-2)]/min_restwindows)
                if n_times_sub == 1:
                    
                    obs=obs_windows[0:(min_restwindows+1)]
                    np.save(os.path.join(path_PD,"observation%02d" % j),obs)
                    leftovers= obs_windows[(min_restwindows+1):-1]
                    if len(leftovers_all_PD)==0:
                        leftovers_all_PD=leftovers
                    else:
                        leftovers_all_PD=np.vstack((leftovers_all_PD, leftovers))
                    
                elif n_times_sub == 2:
                    obs=obs_windows[0:(min_restwindows+1)]
                    np.save(os.path.join(path_PD,"observation%02d" % j),obs)
                    j+=1
                    obs1= obs_windows[(min_restwindows+1):((2*min_restwindows)+1)]
                    np.save(os.path.join(path_PD,"observation%02d" % j),obs1)

                    leftovers=obs_windows[((2*min_restwindows)+1):-1]
                    if len(leftovers_all_PD)==0:
                        leftovers_all_PD=leftovers
                    else:
                        leftovers_all_PD=np.vstack((leftovers_all_PD, leftovers))
        else:
            obs_windows=sub_windows_EMG(ID) #all the rest windows of a sub
            if len(obs_windows)== min_restwindows:
                ID=ID+1
            else: 
                j+=1
                n_times_sub=np.floor(nwindows_rest_all[(ID-1)]/min_restwindows)
                if n_times_sub == 1:
                    
                    obs=obs_windows[0:(min_restwindows+1)]
                    np.save(os.path.join(path_PD,"observation%02d" % j),obs)
                    leftovers= obs_windows[(min_restwindows+1):-1]
                    if len(leftovers_all_PD)==0:
                        leftovers_all_PD=leftovers
                    else:
                        leftovers_all_PD=np.vstack((leftovers_all_PD, leftovers))
                    
                elif n_times_sub == 2:
                    obs=obs_windows[0:(min_restwindows+1)]
                    np.save(os.path.join(path_PD,"observation%02d" % j),obs)
                    j+=1
                    obs1= obs_windows[(min_restwindows+1):((2*min_restwindows)+1)]
                    np.save(os.path.join(path_PD,"observation%02d" % j),obs1)

                    leftovers=obs_windows[((2*min_restwindows)+1):-1]
                    if len(leftovers_all_PD)==0:
                        leftovers_all_PD=leftovers
                    else:
                        leftovers_all_PD=np.vstack((leftovers_all_PD, leftovers))
                    
       
    return leftovers_all_PD
    #my observations are EMG matrices

def create_directories():
    #directory_PD="PD"
    #directory_HC="HC"
    parent_dir='C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/'
    path_parent="Observations"
    os.mkdir(path_parent)
    parent_dir= 'C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Observations/'
    #path_PD = os.path.join(parent_dir, directory_PD)
    #path_HC = os.path.join(parent_dir, directory_HC)
    
    return parent_dir
    
#%%
# def get_data_matrix():
#     data=[]
#     max_restwindows= get_max_restwindows()
#     for ID in range(1,17): 
#         obs_windows_all=[]
#         print(ID)
        
#         if ID==8:
#             ID=ID+1
#         else:
#             if ID >=12: #HC
#                 obs_windows=[]
#                 obs_windows_filled=[]
#                 obs_windows=sub_windows_EMG(ID)
#                 index=0
#                 obs_windows_filled=fill_windows(obs_windows,max_restwindows, index)
                
#             else: #PD
#                 obs_windows=[]
#                 obs_windows_filled=[]
#                 obs_windows=sub_windows_EMG(ID)
#                 index=window_dupli_PD(ID)
#                 obs_windows_filled=fill_windows(obs_windows,max_restwindows, index)
                
                
#             if np.size(data,0)==0:
#                 data = obs_windows_filled
#             else:
#                 data=np.dstack((data,obs_windows_filled))
#         #print(np.size(data,2))
        
#     print("done")
#     return data
 


# def window_dupli_PD(ID): #ID is betwwen 1 and 11
#     labels=check_gyro_tremor(sub_windows_gyro(ID))
#     bag_label_tremor=bag_label(labels) #each sub is a bag
#     index=float("nan")
#     #for i in range
#     if bag_label_tremor==1: #if it is a PD sub
#         i=0
#         while labels[i]==0:
#             i+=1
#             print(i)
#         index=i   
        
#     return bag_label_tremor, index

# def fill_windows(obs_windows,max_restwindows, index): #if the ID is from an HC then index=0, if the ID is from a PD then index=1
#     #obs_windows=sub_windows_EMG(ID)
#     #max_restwindows= get_max_restwindows()
#     if len(obs_windows) < max_restwindows:
#         addition= np.tile(obs_windows[index,:],((int(max_restwindows-len(obs_windows))),1))
#         obs_windows_filled= np.vstack((obs_windows,addition))
#     elif len(obs_windows) == max_restwindows:
#         obs_windows_filled= obs_windows
            
#     return obs_windows_filled

# def bag_label(labels): #labels is an array of 0 and 1, for each PD sub
#     labels=labels.astype(int)
#     bin_count=np.bincount(labels)
#     bag_label_tremor = float("nan")
#     if bin_count[1] >= (round(len(labels)/2)):
#         bag_label_tremor=1
#     return bag_label_tremor #each sub is a bag
 #%% 
def get_tremor_labels(): #gets all the tremor labels(4366)
    labels_tremor_sub=[] #number of tremor labels(positive) of each sub
    labels_all=[]
    for ID in range(1,12): #all PD subjects
        obs_windows=[]
        if ID==8:
            ID=ID+1
        else:
            obs_windows=[]
            obs_windows=sub_windows_gyro(ID)
            labels=check_gyro_tremor(obs_windows)
            for i in range(len(labels)):
                if labels[i]==1:
                    #if len(labels_tremor_sub)==0:
                     #   labels_tremor_sub=np.insert(labels_all,0,1)

                    if len(labels_all)==0:
                        labels_all=np.insert(labels_all,0,1)
                    else:    
                        labels_all=np.insert(labels_all,-1,1)
                else:
                    labels_all=labels_all
        print(ID)
    pos_all=labels_all
    return pos_all
#the negative labels of PD subjects will not be used in the model, only all the negative labels of HC will be considered
#%%

neg_labels=np.zeros(5)
# def get_neg_labels():
#     neg_labels=[]
#     for ID in range(12,17):
#         #obs_windows=[]
#         #obs_windows=sub_windows_EMG(ID)
#         neg_labels=np.append(neg_labels, 0)
        
#         print(ID)
#     return neg_labels
 #%%          
def check_gyro_tremor(obs_gyrowindows_all): #for each sub
    labels=np.zeros(len(obs_gyrowindows_all)) #zeros for initializing
    for i in range(len(obs_gyrowindows_all)): #for each row
     test_row=obs_gyrowindows_all[i,:]
     row_correctmean = test_row - np.mean(test_row)

     #Filter as the EMG
     low=1.5/(2000/2)
     high=8/(2000/2)

     b,a = sp.signal.butter(4, low, btype="highpass")
     d,c = sp.signal.butter(4, high, btype="lowpass") 

     gyro_filtered1 = sp.signal.filtfilt(b, a, row_correctmean) 
     gyro_filtered2 = sp.signal.filtfilt(d,c,gyro_filtered1)
     gyro_norm=(gyro_filtered2-np.mean(gyro_filtered2))/np.std(gyro_filtered2)

     check=bandpower(gyro_norm, 2000, 3.5, 7.5) #should I change the bandpower to 3?
     check_total=bandpower(gyro_norm, 2000, 1.5, 8)
     check_band1=bandpower(gyro_norm, 2000, 1.5, 3.5)
     check_band2=bandpower(gyro_norm, 2000, 7.5, 8)

     if (check/check_total) >= 0.5: #if 50% or more of a window correspond to the band 3.5-7.5 Hz then it is assigned a tremor positive label
     #for each window we have a label, len(labels) is the number of rows of obs_gyrowindows_all of a sub
         labels[i]=1
     else: 
         labels[i]=0
    
    return labels

def bandpower(x, fs, fmin, fmax):
    f, Pxx = signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


#%%
def sub_windows_EMG(ID): #create a matrix of windows of EMG data(each row is a window, and each window is 6000 samples (3s))

    window_labels=get_window_labels(ID, 1)
    #max_restwindows= get_max_restwindows()
    emg_filtered= get_EMG_sub(ID)[1]
    #print(np.shape(emg_filtered))
    
    obs_windows=[]
    window_length=3*2000
    j=0
    
    for i in range(len(window_labels)):
        k = j + window_length
    
        if window_labels[i]==0:
            k = j + window_length
            window_rest=emg_filtered[j:k] 
            #print(len(window_rest))
            if len(obs_windows)==0:
                obs_windows = np.append(obs_windows, window_rest)
            else:
                #print(np.size(obs_windows,1))
                #print(len(window_rest))
                obs_windows = np.vstack((obs_windows, window_rest))
                #print(np.size(obs_windows,1))
        j = k
    # print(obs_windows.shape)
    # if len(obs_windows) < max_restwindows:
    #     addition= np.tile(obs_windows[0,:],((int(max_restwindows-len(obs_windows))),1))
    #     obs_windows_all= np.vstack((obs_windows,addition))
    # elif len(obs_windows) == max_restwindows:
    #     obs_windows_all= obs_windows

    
    #print("done")
    return obs_windows

def sub_windows_gyro(ID): #create a matrix of windows of gyro data (each row is a window, and each window is 6000 samples) (3s))

    window_labels=get_window_labels(ID, 1)
    max_restwindows= get_max_restwindows()
    gyro_x, gyro_y, gyro_z= get_gyro_sub(ID)
    
    gyro_vector_magnitude = np.sqrt(((gyro_x)**2 + (gyro_y)**2 + (gyro_z**2)))

    
    obs_windows=[]
    window_length=3*2000
    j=0
    for i in range(len(window_labels)):
        k = j + window_length
    
        if window_labels[i]==0:
            k = j + window_length
            window_rest=gyro_vector_magnitude[j:k] 
            
            if len(obs_windows)==0:
                obs_windows = np.append(obs_windows, window_rest)
            else:
                obs_windows = np.vstack((obs_windows, window_rest))
                #print(np.size(obs_windows,1))
        j = k
    
    # if len(obs_windows) < max_restwindows:
    #     addition= np.tile(obs_windows[0,:],((int(max_restwindows-len(obs_windows))),1))
    #     obs_gyrowindows_all= np.vstack((obs_windows,addition))
    # if len(obs_windows) == max_restwindows:
    #     obs_gyrowindows_all= obs_windows
    
    #print("done")
    obs_gyrowindows_all=obs_windows
    return obs_gyrowindows_all #, obs_gyrowindows_all

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
            s = 'C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Output Hand classifier/window_labels_ID0{}_T0.025'.format(ID)
            all_files_output = pd.read_csv(s +'.csv')

    else:
        s = 'C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Output Hand classifier/window_labels_ID{}_T0.025'.format(ID) 
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
                print(windows_rest)
            nwindows_rest_all=np.append(nwindows_rest_all,windows_rest)
        
            max_restwindows= np.max(nwindows_rest_all)
    
    return  max_restwindows 

#%%
def get_min_restwindows(): 
    nwindows_rest_all=[]

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

            min_restwindows= np.min(nwindows_rest_all)
            
    return  nwindows_rest_all, min_restwindows



#%% Save data matrix

# Some test data

def save_mat_data(data):
    data=get_data_matrix()
    matfile = 'data_matrix.mat'
# Specify the filename of the .mat file
    # Write the array to the mat file. For this to work, the array must be the value
    # corresponding to a key name of your choice in a dictionary
    scipy.io.savemat(matfile, mdict={'out': data}, oned_as='row')

def load_mat_data():
    matfile = 'data_matrix.mat'
    #data=get_data_matrix()
    # Now load in the data from the .mat that was just saved
    matdata = scipy.io.loadmat(matfile)
    # And just to check if the data is the same:

    #assert np.all(data == matdata['out'])      
    data = matdata['out']
    return data
        