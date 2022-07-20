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
 #Subject ID

def check_recordings_meta(file_meta):
    data_X=file_meta.values[:,0]
    for j in range(len(data_X)):
        if data_X[j]== "Recalibration Times:":
            a=j
            timestamps=data_X[3:a]
    timestamps=timestamps.astype("float64")
    indexes=timestamps*fs
       
    return timestamps,indexes
#%%

all_files=get_all_data()
#%%
ID=10
timestamps,indexes=check_recordings_meta(all_files[(ID-1)]) #meta file number is the subject ID -1
    
#%% Retrieve the EMG,Accelerometer and Gyroscope data for one subject

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
emg, acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z=check_recordings_data(all_files[(ID+15)]) #file number is the subject ID +15

 
 #%% 
 
def plt_time_emg(ID,emg,timestamps):
    #Time domain plot
    plt.figure()
    for i in range(len(timestamps)) :
       x=timestamps[i]
       #x1=
       plt.axvline(x, color="#6F1E51") #magenta purple
    time2 = np.arange(0,len(emg)/2000, 1/2000) # sampling rate 2000 Hz

    plt.plot(time2,emg,  color="#5758BB") #circumorbital ring
    plt.show()
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    plt.title(ID)

def plt_time_acce(ID,timestamps, x,y,z):
    plt.figure()
    for i in range(len(timestamps)) :
        x1=timestamps[i]
        plt.axvline(x1, color="blue")
    time = np.arange(0,len(emg)/2000, 1/2000) 
    plt.plot(time,x, label= "X")
    plt.plot(time,y, label= "Y")
    plt.plot(time,z, label="Z")
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
    plt.plot(frequency, power_spectrum,color="#5758BB")
    plt.ylim([-5000, 5.5e9])

    #plt.ylim([-5000, 1.5e6])
    plt.xlim([-2, 120])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title(ID)
    
#%%    For accelerometer and gyroscope

def plt_freq(data, ID):
    fs=2000
    data=data-np.mean(data)
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, fs/2, len(power_spectrum))
    
    plt.figure()
    plt.plot(frequency, power_spectrum,color="#5758BB")
    #plt.ylim([-5000000, 4.5e9])

    plt.ylim([-50000, 1.5e7])
    plt.xlim([-0.5, 10])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')

    plt.title(ID)
    


#%%

def filtering_emg_alt(emg):
    fs=2000    
    
    # process EMG signal: remove mean
    emg_correctmean = emg - np.mean(emg)
    
    
    # create bandpass filter for EMG
    low = 1.5/(fs/2)
    high = 8/(fs/2)
    
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
    time2 = np.arange(0,len(emg)/2000, 1/2000) 
     # plot comparison of EMG with offset vs mean-corrected values
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Mean offset present')
    plt.plot(time2,emg,color="#5758BB")
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg), np.amax(emg))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Mean-corrected values')
    plt.plot( time2,emg_correctmean,color="#5758BB")
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_correctmean), np.amax(emg_correctmean))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    
    # plot comparison of unfiltered vs filtered mean-corrected EMG
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unfiltered EMG')
    plt.plot(time2, emg_correctmean,color="#5758BB")
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_correctmean), np.amax(emg_correctmean))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Filtered EMG')
    plt.plot(time2, emg_filtered,color="#5758BB")
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(np.amin(emg_filtered), np.amax(emg_filtered))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')

    # plot comparison of Unnormalized(after filtering) vs Normalized EMG
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unstandardized EMG')
    plt.plot(time2,emg_filtered,color="#5758BB")
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(np.amin(emg_filtered), np.amax(emg_filtered))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')
    
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Standardization EMG')
    plt.plot(time2,emg_normalized,color="#5758BB")
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(np.amin(emg_normalized), np.amax(emg_normalized))
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (V)')

