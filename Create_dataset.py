from cmath import nan
import math
import os
from random import sample
from shutil import copyfile

import numpy as np
import pandas as pd
import scipy as sp
import scipy.io
from scipy import signal

from preProcessing import (check_recordings_data, filtering_emg_alt,
                           get_all_data)



#the order is all PD, all HC + leftovers HC, leftovers PD, resample HC


def save_file(obs, group, j):
    """ Saves a file in the Observations Folder, assigning it the ID and label """
    if (group=="HC"):
        bag_label_tremor = 0
    else:
        bag_label_tremor = check_gyro_tremor(obs)      
    np.save("./Observations/observation%02d_%d" % (j, bag_label_tremor), obs)


    
    
def obs_leftovers_HC(leftovers_all_HC,min_restwindows):
    n_times_HC=math.floor(len(leftovers_all_HC)/min_restwindows) #it will be 1
    obs=leftovers_all_HC[0:(min_restwindows)]
    path="./Observations"
    dirs = os.listdir( path )
    save_file(obs,"HC", len(dirs)+1)
    
    
def obs_leftovers_PD(leftovers_all_PD,min_restwindows):
    path="./Observations"
    dirs = os.listdir( path )
    n_times_PD=math.floor(len(leftovers_all_PD)/min_restwindows) #it will be 4
    j = len(dirs)+1
    for i in range(n_times_PD):
        obs = leftovers_all_PD[min_restwindows*i:(min_restwindows*(i+1))]
        save_file(obs,"PD",j)
        j += 1
    

def create_observations():
    if not os.path.exists("./Observations"):
        os.mkdir("./Observations")

    
    min_restwindows=get_min_restwindows()[1]
    min_restwindows = int(min_restwindows)
    leftovers_all_HC, leftovers_all_PD = create_observations_main(min_restwindows)
    create_leftovers(leftovers_all_HC, leftovers_all_PD, min_restwindows)
    resampleHC()
    return 0


def create_leftovers(leftovers_all_HC, leftovers_all_PD, min_restwindows):
    min_restwindows = int(min_restwindows)
    obs_leftovers_HC(leftovers_all_HC,min_restwindows)
    obs_leftovers_PD(leftovers_all_PD,min_restwindows)
    return 0

def resampleHC(): 
    file = "./Observations/observation%02d_0.npy"
    randomfiles = sample(range(13,22),6)
    print(randomfiles)
    for i in range(27,33):
        copyfile(file % randomfiles[i-27], file % i )

def create_observations_main(min_restwindows):
    #min_restwindows=get_min_restwindows()[1]
    #min_restwindows = int(min_restwindows)
    leftovers_all_HC=[]
    leftovers_all_PD=[]
    j=0
    for ID in range(1,17):
        print(ID)
        if ID == 8:
            continue
        
        obs_windows=sub_windows_EMG(ID) #all the rest windows of a sub
        total_len = len(obs_windows) + 1
    
        i = 0  
        while (min_restwindows*(i+1) <= total_len):
            j+=1
            obs=obs_windows[min_restwindows*i:min_restwindows*(i+1)]
    
            if ID in range(12,17):
                save_file(obs,"HC",j)
            else:
                save_file(obs,"PD",j)
            i+=1
    
        leftovers = obs_windows[min_restwindows*i:]
        if ID in range(12,17):
            if len(leftovers_all_HC)==0:
                leftovers_all_HC=leftovers
            else:
                leftovers_all_HC=np.vstack((leftovers_all_HC, leftovers))
        else:
            if len(leftovers_all_PD)==0:
                leftovers_all_PD=leftovers
            else:
                leftovers_all_PD=np.vstack((leftovers_all_PD, leftovers))
    
    return leftovers_all_HC, leftovers_all_PD


 
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
     #check_band1=bandpower(gyro_norm, 2000, 1.5, 3.5)
     #check_band2=bandpower(gyro_norm, 2000, 7.5, 8)
     
     

     if (check/check_total) >= 0.5: #if 50% or more of a window correspond to the band 3.5-7.5 Hz then it is assigned a tremor positive label
     #for each window we have a label, len(labels) is the number of rows of obs_gyrowindows_all of a sub
         labels[i]=1
     else: 
         labels[i]=0
    
    bag_label_tremor = np.sum(labels) >= len(labels)/2
    return bag_label_tremor



def bandpower(x, fs, fmin, fmax):
    f, Pxx = signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])



def sub_windows_EMG(ID): #create a matrix of windows of EMG data(each row is a window, and each window is 6000 samples (3s))

    window_labels=get_window_labels(ID, 1)
    emg_filtered= get_EMG_sub(ID)[1]
    
    obs_windows=[]
    window_length=3*2000
    j=0
    
    for i in range(len(window_labels)):
        k = j + window_length
    
        if window_labels[i]==0:
            k = j + window_length
            window_rest=emg_filtered[j:k] 
            if len(obs_windows)==0:
                obs_windows = np.append(obs_windows, window_rest)
            else:
                obs_windows = np.vstack((obs_windows, window_rest))
        j = k

    return obs_windows

def sub_windows_gyro(ID): #create a matrix of windows of gyro data (each row is a window, and each window is 6000 samples) (3s))

    window_labels=get_window_labels(ID, 1)
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
        j = k
    
    obs_gyrowindows_all=obs_windows
    return obs_gyrowindows_all


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
        
        for i in range(len(windowtemp_labels)):
            window_labels=np.append(window_labels,windowtemp_labels[i][0])
            
        return window_labels


def get_max_restwindows(): 
    nwindows_rest_all=[]

    for ID in range(1,17):

        if ID==8:
            ID=ID+1
        else:
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


def get_min_restwindows(): 
    nwindows_rest_all=[]

    for ID in range(1,17):
        
        if ID==8:
            ID=ID+1
        else:
            window_labels=get_window_labels(ID, 1)
            nwindows_rest=[]
            for j in range(len(window_labels)):
                if int(window_labels[j])==0:
                    nwindows_rest= np.append(nwindows_rest,window_labels[j])
                windows_rest=len(nwindows_rest)

            nwindows_rest_all=np.append(nwindows_rest_all,windows_rest)

            min_restwindows= np.min(nwindows_rest_all)
            
    return  nwindows_rest_all, min_restwindows


#%%
# def main():
#     create_observations()
#     return 0


# if __name__ == "__main__":
#     main()