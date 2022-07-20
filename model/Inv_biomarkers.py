
import numpy as np
import pandas as pd
from dataloader import ParkinsonsDataset
#from Create_dataset import get_window_labels
from preProcessing import (check_recordings_data, filtering_emg_alt,
                           get_all_data)
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

#obs1=np.load("../utils/Observations/observation01_1.npy")
#obs2= np.load("../utils/Observations/observation02_1.npy")

ID=1

#window_labels = get_window_labels(1,1)

def get_acce_sub(ID):
    
    file_data= get_all_data()[(ID+15)]
    Acce_X,Acce_Y, Acce_Z= check_recordings_data(file_data)[1], check_recordings_data(file_data)[2],check_recordings_data(file_data)[3]
    return Acce_X,Acce_Y, Acce_Z

def sub_windows_acce(ID): #create a matrix of windows of gyro data (each row is a window, and each window is 6000 samples) (3s))

    window_labels=get_window_labels(ID, 1)
    acce_x, acce_y, acce_z= get_acce_sub(ID)
    
    #acce_vector_magnitude = np.sqrt(((acce_x)**2 + (acce_y)**2 + (acce_z**2)))

    
    obs_windows_x=[]
    obs_windows_y=[]
    obs_windows_z=[]
    window_length=3*2000
    j=0
    for i in range(len(window_labels)):
        k = j + window_length
    
        if window_labels[i]==0:
            k = j + window_length
            window_rest_x=acce_x[j:k] 
            window_rest_y =acce_y[j:k]
            window_rest_z=acce_z[j:k]
            
            if len(obs_windows_x)==0:
                obs_windows_x = np.append(obs_windows_x, window_rest_x)
                obs_windows_y = np.append(obs_windows_y, window_rest_y)
                obs_windows_z = np.append(obs_windows_z, window_rest_z)
            else:
                obs_windows_x = np.vstack((obs_windows_x, window_rest_x))
                obs_windows_y = np.vstack((obs_windows_y, window_rest_y))
                obs_windows_z = np.vstack((obs_windows_z, window_rest_z))
        j = k
    
    #obs_accewindows_all=np.dstack((obs_windows_x, obs_windows_y,obs_windows_z))
    #return obs_accewindows_all #for the test observations (belong to ID1), the matrix has 870 rows
    return obs_windows_x, obs_windows_y,obs_windows_z

obs_windows_x, obs_windows_y,obs_windows_z=sub_windows_acce(1)

d={"x":[obs_windows_x],"y":[obs_windows_y], "z":[obs_windows_z]}
df=pd.DataFrame(d)
#%% 
def sub_windows_acce1(ID): #create a matrix of windows of gyro data (each row is a window, and each window is 6000 samples) (3s))

    window_labels=get_window_labels(ID, 1)
    acce_x, acce_y, acce_z= get_acce_sub(ID)
    
    acce_vector_magnitude = np.sqrt(((acce_x)**2 + (acce_y)**2 + (acce_z**2)))

    
    obs_windows=[]
    window_length=3*2000
    j=0
    for i in range(len(window_labels)):
        k = j + window_length
    
        if window_labels[i]==0:
            k = j + window_length
            window_rest=acce_vector_magnitude[j:k] 
            
            if len(obs_windows)==0:
                obs_windows = np.append(obs_windows, window_rest)
            else:
                obs_windows = np.vstack((obs_windows, window_rest))
        j = k
    
    obs_accewindows_all=obs_windows
    return obs_accewindows_all

obs_accewindows_all_2=sub_windows_acce1(1)

#%% 
def get_instance(data, A):
    largeAttention= data[torch.argmax(A)]
    smallAttention = data[torch.argmin(A)]
    
    largest_instance=torch.argmax(largeAttention)
    smallest_instance=torch.argmin(smallAttention)
    
    #large_acc_win=data[largest_instance]
    #small_acc_win=data[smallest_instance]
    return  largest_instance, smallest_instance
    
    
    

def power_acce(instance): #instance is the instance with the highest attention
    fs=2000
    data_mean=instance-np.mean(instance)
    
    fourier_transform = np.fft.rfft(data_mean)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, fs/2, len(power_spectrum))
    
    plt.figure()
    plt.plot(frequency, power_spectrum,color="#5758BB")
    plt.xlim([-0.5,10])
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Power")

#power_acce(obs_windows_y[327])
#power_acce(obs_accewindows_all[19])

#hig 53
#low 56

#%%
def auto_acce(data):
    autocorrelation=sm.tsa.acf(data)
    
    tsaplots.plot_acf(data)

#%% 

time2 = np.arange(0,6000/2000, 1/2000) 

plt.figure()    
plt.plot(time2,obs_windows_x[327], color="#5758BB", label="x")
plt.plot(time2, obs_windows_y[327], color= "#1289A7",label="y")
plt.plot(time2,obs_windows_z[327],color= "#FDA7DF", label="z")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("G")

#%% 

time2 = np.arange(0,6000/2000, 1/2000) 

plt.figure()    
plt.plot(time2,obs_windows_x[327], color="#5758BB", label="x")
plt.plot(time2, obs_windows_y[327], color= "#1289A7",label="y")
plt.plot(time2,obs_windows_z[327],color= "#FDA7DF", label="z")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("G")