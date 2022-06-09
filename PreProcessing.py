# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:49:28 2022

@author: mhele
"""
import csv
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
from scipy import signal
from scipy.signal import butter, lfilter, hilbert, chirp
from scipy.signal import spectrogram

#%% Create a DataFrame with all the csv files of the subjects' recordings 


def get_all_data():
    paths = glob.glob("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Converted_csv/*.csv")
    
    all_files=[]
    for filename in paths:
        df= pd.read_csv(filename, index_col=None, header=0)
        all_files.append(df)
    return all_files

  
#%% Retreive the timestamps of the start and end of a task for one subject
fs=2000
ID=12 #Subject ID

def check_recordings_meta(file_meta):
    data_X=file_meta.values[:,0]
    for j in range(len(data_X)):
        if data_X[j]== "Recalibration Times:":
            a=j
            timestamps=data_X[3:a]
    timestamps=timestamps.astype("float64")
    indexes=timestamps*fs
       
    return indexes
#%%
#all_files=get_all_data()
#indexes=check_recordings_meta(all_files[(ID-1)]) #meta file number is the subject ID -1
    
#%% Retreive the EMG,Accelerometer and Gyroscope data for one subject

def check_recordings_data(file_data):
    data=file_data.values
    EMGSignal=data[:,1] * (3.3/8191) #volts 
    Accelerometer_X=data[:,6] * (1/8192) #gravitational constants
    Accelerometer_Y=data[:,7] * (1/8192)
    Accelerometer_Z=data[:,8] * (1/8192)
    
    Gyro_X=data[:,3] * (1/16.384) #degrees per second
    Gyro_Y=data[:,4] * (1/16.384)
    Gyro_Z=data[:,5] * (1/16.384)
    
    return EMGSignal, Accelerometer_X, Accelerometer_Y, Accelerometer_Z, Gyro_X,Gyro_Y, Gyro_Z

#%%
#emg, acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z=check_recordings_data(get_all_data()[(ID+15)]) #file number is the subject ID +15

 
 #%% 
 
def plt_time_emg(emg,indexes):
    #Time domain plot
    plt.figure()
    for i in range(len(indexes)) :
       x=indexes[i]
       plt.axvline(x, color="blue")
       
    plt.plot(emg, 'r-')
    plt.show()
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    plt.title(ID)

def plt_time_acce(indexes, x,y,z):
    plt.figure()
    for i in range(len(indexes)) :
        x1=indexes[i]
        plt.axvline(x1, color="blue")
       
    plt.plot(x, label= "X")
    plt.plot(y, label= "Y")
    plt.plot(z, label="Z")
    plt.legend(loc="best")
    plt.show()
    plt.xlabel('Time (s)')
    plt.ylabel('G')
    plt.title(ID)

#%%
def plt_freq_emg(emg, ID):  
    
    emg_correctmean = emg - np.mean(emg)
    emg_normalized= (emg_correctmean- np.mean(emg_correctmean))/np.std(emg_correctmean)

    
    fourier_transform = np.fft.rfft(emg_normalized)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, fs/2, len(power_spectrum))
    
    plt.figure()
    plt.plot(frequency, power_spectrum)
    #plt.ylim([-5000000, 4.5e9])

    #plt.ylim([-500000, 1.5e7])
    #plt.xlim([-2, 70])
    plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Amplitude')

    plt.title(ID)
#%%    For accelerometer and gyroscope

def plt_freq(data, ID):
    fs=500
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, fs/2, len(power_spectrum))
    
    plt.figure()
    plt.plot(frequency, power_spectrum)
    #plt.ylim([-5000000, 4.5e9])

    #plt.ylim([-500000, 1.5e7])
    #plt.xlim([-2, 70])
    plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Amplitude')

    plt.title(ID)
    


#%%

def filtering_emg_alt(emg):
    fs=2000    
    
    # process EMG signal: remove mean
    emg_correctmean = emg - np.mean(emg)
    
    
    # create bandpass filter for EMG
    low = 1.5/(2000/2)
    high = 8/(2000/2)
    
    b,a = sp.signal.butter(4, low, btype="highpass")
    d,c = sp.signal.butter(4, high, btype="lowpass") 
    
    
    
    # Plot magnitude response of the filter
    #w, h = signal.freqz(b,a)
    #Mag = 20*np.log10(abs(h))
    #Freq = w*fs/(2*np.pi)
    
    
#    plt.figure()
#    plt.plot(Freq, Mag, 'r')
#    plt.title('Digital Magnitude Response')
#    plt.xlabel('Frequency [Hz]')
#    plt.ylabel('Magnitude [dB]')
#    plt.grid()
#    
#    wz, hz = signal.freqs(b, a)
#    plt.figure()
#    plt.semilogx(wz, 20 * np.log10(abs(hz)))
#    plt.title('Analog Magnitude Response')
#    plt.xlabel('Frequency')
#    plt.ylabel('Amplitude response [dB]')
#    plt.grid()
#    plt.show()
#    
#    # Calculate phase angle in degree from hz
#    Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))*(180/np.pi)
#    plt.figure()
#    plt.plot(Freq, Phase, 'g')
#    plt.title(r'Phase response')
#    plt.xlabel('Frequency [Hz]')
#    plt.ylabel('Phase (degree)')
#    plt.grid()
    
    
    
    # process EMG signal: filter EMG
    emg_filtered1 = sp.signal.filtfilt(b, a, emg_correctmean) 
    emg_filtered = sp.signal.filtfilt(d,c,emg_filtered1)
    
    #Z-score normalization
    emg_normalized= (emg_filtered- np.mean(emg_filtered))/np.std(emg_filtered)
    
    return emg_correctmean,emg_filtered,emg_normalized



#%%
def plot_filtering(emg, emg_correctmean,emg_filtered,emg_normalized):
    
     # plot comparison of EMG with offset vs mean-corrected values
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Mean offset present')
    plt.plot(emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg), np.amax(emg))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Mean-corrected values')
    plt.plot( emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_correctmean), np.amax(emg_correctmean))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    
    # plot comparison of unfiltered vs filtered mean-corrected EMG
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unfiltered EMG')
    plt.plot(emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_correctmean), np.amax(emg_correctmean))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Filtered EMG')
    plt.plot(emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_filtered), np.amax(emg_filtered))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')

    # plot comparison of Unnormalized(after filtering) vs Normalized EMG
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unnormalized EMG')
    plt.plot(emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_filtered), np.amax(emg_filtered))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Normalized EMG')
    plt.plot(emg_normalized)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_normalized), np.amax(emg_normalized))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')


#%% Second attempt at filtering
#emg_correctmean,emg_filtered,emg_normalized=filtering_emg_alt(emg)


#%%Normalization accelerometer

# accel= np.zeros((len(acce_x),3))

# #file=open("./Mean and std.txt", "r")
# #stat=np.array(file.readlines(), dtype="float64") #not working
# stat=np.array([0.056397206840420475,0.15522436067451023,0.9160927112397802,0.1464625741042218,0.3217536296020726,0.16548120097821328])

# acce_x_norm= (acce_x- stat[0])/stat[3]
# acce_y_norm= (acce_y- stat[1])/stat[4]
# acce_z_norm= (acce_z- stat[2])/stat[5]

# plt.figure()
# plt.plot(Accelerometer_X, label= "X")
# plt.plot(Accelerometer_Y, label= "Y")
# plt.plot(Accelerometer_Z, label="Z")
# plt.title("Acce id2 dataframe")

# #%%
# plt.figure()
# plt.plot(Gyro_X, label= "X")
# plt.plot(Gyro_Y, label= "Y")
# plt.plot(Gyro_Z, label="Z")
# plt.title("Gyro id2 dataframe")


