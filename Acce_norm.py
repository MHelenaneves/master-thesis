import csv
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% Create a DataFrame with all the csv files of the subjects' recordings 

paths = glob.glob("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/Code/Converted_csv/*.csv")

all_files=[]
for filename in paths:
    df= pd.read_csv(filename, index_col=None, header=0)
    all_files.append(df)
    
#%% Extract the Accelerometer data from each subject's csv data file

def check_recordings_data(file_data):
    data=file_data.values
    Accelerometer_X=data[:,6] * (1/8192) #gravitational constants
    Accelerometer_Y=data[:,7] * (1/8192)
    Accelerometer_Z=data[:,8] * (1/8192)
    
    return  Accelerometer_X, Accelerometer_Y, Accelerometer_Z
    
#%% Z score normalization of accelerometer data across all subjects

def norm_Acce(all_files):

    Acce_x=[]
    Acce_y=[]
    Acce_z=[]

    for i in range(16):
        acce_x, acce_y, acce_z=check_recordings_data(all_files[(i+16)]) #I have 16 subjects
        Acce_x=np.append(Acce_x,acce_x)
        Acce_x=np.append(Acce_x,acce_x)
        Acce_y=np.append(Acce_y,acce_y)
        Acce_z=np.append(Acce_z,acce_z)

    Accex_mean=np.mean(Acce_x)
    Accey_mean=np.mean(Acce_y)
    Accez_mean=np.mean(Acce_z)
    
    Accex_std=np.std(Acce_x)
    Accey_std=np.std(Acce_y)
    Accez_std=np.std(Acce_z)
    
    return Accex_mean, Accey_mean, Accez_mean, Accex_std, Accey_std, Accez_std

#%%
Accex_mean, Accey_mean, Accez_mean, Accex_std, Accey_std, Accez_std=norm_Acce(all_files)

#%%
file=open("Mean and std.txt", "w")
file.write(repr(Accex_mean) + "\n" + repr(Accey_mean) + "\n" + repr(Accez_mean) + "\n" + repr(Accex_std) + "\n" + repr(Accey_std) + "\n" + repr(Accez_std))
file.close

