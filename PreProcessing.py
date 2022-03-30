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

#%%

paths = glob.glob("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/Code/Converted_csv/*.csv")

all_files=[]
for filename in paths:
    df= pd.read_csv(filename, index_col=None, header=0)
    all_files.append(df)

        
#%%
fs=2000

def check_recordings_meta(file_meta):
    data_X=file_meta.values[:,0]
    for j in range(len(data_X)):
        if data_X[j]== "Recalibration Times:":
            a=j
            timestamps=data_X[3:a]
    timestamps=timestamps.astype("float64")
    indexes=timestamps*fs
       
    return data_X, timestamps,indexes

            
data_X,timestamps,indexes=check_recordings_meta(all_files[6])
    

#for i in range(len(all_files)):


 #%% 
 
def plt_time_emg(emg,indexes):   
    #Time domain plot
    plt.figure()
    for i in range(len(indexes)) :
       x=indexes[i]
       plt.axvline(x, color="blue")
       
    plt.plot(emg, 'r-')
    plt.show()
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (mV)')

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
    plt.xlabel('Time (sec)')

def plt_freq_emg(emg):   
    fourier_transform = np.fft.rfft(emg)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, fs/2, len(power_spectrum))
    
    plt.figure()
    plt.plot(frequency, power_spectrum)
    
    plt.figure()
    plt.magnitude_spectrum(emg, fs, scale='dB', color='C1')

#%%

def check_recordings_data(file_data,indexes):
    data=file_data.values
    EMGSignal=data[:,1]
    Accelerometer_X=data[:,6]
    Accelerometer_Y=data[:,7]
    Accelerometer_Z=data[:,8]
    
    #return EMGSignal, Accelerometer_X, Accelerometer_Y, Accelerometer_Z, plt_time_emg(EMGSignal, indexes), plt_time_acce(indexes, Accelerometer_X,Accelerometer_Y,Accelerometer_Z)
    return EMGSignal, Accelerometer_X, Accelerometer_Y, Accelerometer_Z

#emg, acce_x, acce_y, acce_z, emg_plot, acce_plot=check_recordings_data(all_files[7],indexes)
emg, acce_x, acce_y, acce_z=check_recordings_data(all_files[13],indexes)

#plt_freq_emg(emg)

#%%

def filtering_emg(emg):    
    
    # process EMG signal: remove mean
    emg_correctmean = emg - np.mean(emg)
    
    
    # create bandpass filter for EMG
    low = 3/(2000/2)
    high = 60/(2000/2)
    b, a = sp.signal.butter(4, [low,high], btype='bandpass')
    
    lim_low=49/(2000/2)
    lim_high= 51/(2000/2)
    
    d, c=sp.signal.butter(4, [lim_low, lim_high], btype="bandstop") #notch at 50 Hz, influence from surrounding electronics
    
    # process EMG signal: filter EMG
    emg_filtered1 = sp.signal.filtfilt(b, a, emg_correctmean) #bandpass
    emg_filtered = sp.signal.filtfilt(d,c, emg_filtered1) #notch
    
    
    #Z-score normalization
    emg_normalized= (emg_filtered- np.mean(emg_filtered))/np.std(emg_filtered)
    
    return emg, emg_correctmean,emg_filtered,emg_normalized

#%%
def plot_filtering(emg, emg_correctmean,emg_filtered,emg_normalized):
    
     # plot comparison of EMG with offset vs mean-corrected values
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Mean offset present')
    #plt.plot(time_cut, emg_cut)
    plt.plot(emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg), np.amax(emg))
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Mean-corrected values')
    #plt.plot(time_cut, emg_correctmean)
    plt.plot( emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_correctmean), np.amax(emg_correctmean))
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    # plot comparison of unfiltered vs filtered mean-corrected EMG
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unfiltered EMG')
    #plt.plot(time_cut, emg_correctmean)
    plt.plot(emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_correctmean), np.amax(emg_correctmean))
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Filtered EMG')
    #plt.plot(time_cut, emg_filtered)
    plt.plot(emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_filtered), np.amax(emg_filtered))
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

# plot comparison of Unnormalized(after filtering) vs Normalized EMG
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unnormalized EMG')
    #plt.plot(time_cut, emg_filtered)
    plt.plot(emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_filtered), np.amax(emg_filtered))
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Normalized EMG')
    #plt.plot(time_cut, emg_normalized)
    plt.plot(emg_normalized)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_normalized), np.amax(emg_normalized))
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')


#%%

emg, emg_correctmean,emg_filtered,emg_normalized=filtering_emg(emg)
#plot_filtering(emg, emg_correctmean,emg_filtered,emg_normalized)
#plt_time_emg(emg_normalized, indexes)
#plt_freq_emg(emg_normalized)

#%%
# plt.figure()
# f, t, Sxx = signal.spectrogram(emg, fs)
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

plt.figure()
plt.specgram(emg,fs)
#%%
plt.figure()
plt.specgram(emg_normalized,fs)

