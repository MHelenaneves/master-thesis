# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:52:44 2022

@author: mhele
"""
#%% Isolate rest windows

nwindows_rest=[]

for i in range(len(window_labels)):
    if window_labels[i]==0:
        nwindows_rest= np.append(nwindows_rest,window_labels[i])

#%% Check data

def plt_steps(indexes,x,y,z,accelerometer_vector_magnitude,accelerometer_vector_magnitude_filt,rolling_cov,window_labels):
    plt.figure()
    
    plt.subplot(5, 1, 1)
    for i in range(len(indexes)) :
       x1=indexes[i]
       plt.axvline(x1, color="blue")
    plt.plot(x, label= "X")
    plt.plot(y, label= "Y")
    plt.plot(z, label="Z")
    plt.legend(loc="best")
    
    plt.subplot(5,1,2)
    plt.plot(accelerometer_vector_magnitude)
    
    plt.subplot(5,1,3)
    plt.plot(accelerometer_vector_magnitude_filt)
    
    plt.subplot(5,1,4)
    plt.plot(rolling_cov)
    
    plt.subplot(5,1,5)
    plt.plot(window_labels)
    
    
plt_steps(indexes,accel[:,0],accel[:,1], accel[:,2,],accelerometer_vector_magnitude,accelerometer_vector_magnitude_filt,rolling_cov,window_labels2)

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
plt.axhline(y=0.025, color='r', linestyle='-')
plt.plot(rolling_cov)


