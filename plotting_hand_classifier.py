# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:52:44 2022

@author: mhele
"""
import csv
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% Isolate rest windows

nwindows_rest=[]

for i in range(len(window_labels)):
    if window_labels[i]==0:
        nwindows_rest= np.append(nwindows_rest,window_labels[i])

#%% Check data

def plt_steps(indexes,x,y,z,accelerometer_vector_magnitude,accelerometer_vector_magnitude_filt,rolling_cov,window_labels):
    time2 = np.arange(0,len(x)/2000, 1/2000)
    plt.figure()
    
    plt.subplot(5, 1, 1)
    for i in range(len(indexes)) :
       x1=indexes[i]
       plt.axvline(x1, color="#6F1E51")
    plt.plot(x, label= "X",color="#5758BB")
    plt.plot(y, label= "Y",color="#1B1464")
    plt.plot(z, label="Z",color="#006266")
    plt.ylabel("G")
    
    plt.legend(loc="best")
    
    plt.subplot(5,1,2)
    plt.plot(accelerometer_vector_magnitude,color="#5758BB")
    plt.ylabel("G")
    
    plt.subplot(5,1,3)
    plt.plot(accelerometer_vector_magnitude_filt,color="#5758BB")
    plt.ylabel("G")
    
    plt.subplot(5,1,4)
    plt.plot(rolling_cov,color="#5758BB")
    plt.ylabel("CV")
    
    plt.subplot(5,1,5)
    plt.plot(window_labels,color="#5758BB")
    
    
plt_steps(indexes,acce_x,acce_y, acce_z,accelerometer_vector_magnitude,accelerometer_vector_magnitude_filt,rolling_cov,window_labels)

#%% Find the threshold (run this section before the previous one if I haven't ran "PreProcessing.py")

paths = glob.glob("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/Code/Converted_csv/*.csv")

all_files=[]
for filename in paths:
    df= pd.read_csv(filename, index_col=None, header=0, low_memory=False)
    all_files.append(df)
#%%
ID=10
fs=2000

def check_recordings_meta(file_meta):
    data_X=file_meta.values[:,0]
    for j in range(len(data_X)):
        if data_X[j]== "Recalibration Times:":
            a=j
            timestamps=data_X[3:a]
    timestamps=timestamps.astype("float64")
    indexes=timestamps*fs
       
    return indexes

            
indexes=check_recordings_meta(all_files[(ID-1)])

#%%  Plot raw accelerometer data
plt.figure()

for i in range(len(indexes)) :
   x1=indexes[i]
   plt.axvline(x1, color="blue")
   
plt.plot(wrist_accel[:,0], label= "X")
plt.plot(wrist_accel[:,1], label= "Y")
plt.plot(wrist_accel[:,2], label="Z")
plt.legend(loc="best")  
plt.title(ID)
plt.xlabel("Time(s)")
plt.ylabel("Acceleration (G)")


#%% Plot normalized accelerometer data

plt.figure()

for i in range(len(indexes)) :
   x1=indexes[i]
   plt.axvline(x1, color="blue")
   
plt.plot(accel[:,0], label= "X")
plt.plot(accel[:,1], label= "Y")
plt.plot(accel[:,2], label="Z")
plt.legend(loc="best")
plt.title(ID)
plt.xlabel("Time(s)")
plt.ylabel("Acceleration (G)")
#plt.ylim([-8, 8])


#%% How to pick the threshold? By visual inspection/ determined empirically

plt.figure()
#for i in range(len(indexes)) :
 #  x1=indexes[i]
  # plt.axvline(x1, color="blue")
time2 = np.arange(0,len(rolling_cov)/2000, 1/2000)
plt.axhline(y=0.025, color="#ED4C67", linestyle='-')
plt.plot(rolling_cov,color="#5758BB")
plt.ylabel("G")
plt.xlabel("Time (s)")

#%% Create bar plot

def get_win_data():
    paths = glob.glob("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Output Hand classifier/*.csv")
    
    all_win_files=[]
    for filename in paths:
        df= pd.read_csv(filename, index_col=None, header=0)
        all_win_files.append(df)
    all_win_files=all_win_files[16:32]
    return all_win_files

#all_win_files=get_win_data()

def get_total_win():
    #all_win_files=get_win_data()
    win_total=[]
    n_rest=[]
    for i in range(17):
        index=i-1
        win_total=np.append(win_total,len(all_win_files[index]))
        
        nwindows_rest=[]
        window_labels=(all_win_files[index]).values[:,0]
        for i in range(len(all_win_files[index])):
            if window_labels[i]==0:
                nwindows_rest= np.append(nwindows_rest,window_labels[i])
        n_rest=np.append(n_rest,len(nwindows_rest))
        
            
    return win_total, n_rest

def bar_plot():
    win_total, n_rest=get_total_win()
    win_total=np.delete(win_total[0:16],7)
    n_rest=np.delete(n_rest[0:16],7)
    data=np.vstack((win_total[0:16],n_rest[0:16]))
    barWidth = 0.35
    fig=plt.subplots()
    br1 = np.arange(15)
    br2 = [x + barWidth for x in br1]
    fig=plt.subplots()
    #ax = fig.add_axes([0,0,1,1])
    plt.bar(br1, data[0], color = "#1B1464", width = barWidth, label= "Total nº windows")
    plt.bar(br2 , data[1], color = '#B53471', width = barWidth, label="Rest windows")    
    plt.ylabel("Nº of windows")
    plt.xlabel("ID")
    plt.xticks([r + barWidth for r in range(15)],
        ['1', '2', '3', '4', '5', "6","7","9","10","11","12","13","14","15","16"])
    plt.legend()
    plt.show()

    
