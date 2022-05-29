'''
This file contains code to run a hand movement classifier based on accelerometer data obtained from a wearable
device located on the forearm.
'''

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import glob

#%% Load the accelerometer data of all subjects
def init():
    paths = glob.glob("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Converted_csv/*.csv")
    
    all_files=[]
    for filename in paths:
        df= pd.read_csv(filename, index_col=None, header=0, low_memory=False)
        all_files.append(df)
    
    return all_files


def main(ID, init):
        
    #%% For every subject (except ID8 which was excluded from the study)
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
    
    
    def compute_rolling_mean(x, window_length):
        '''
        Method to compute rolling mean.
        :param x: 1D numpy array
        :param window_length: Length of window for computing rolling mean. Must be an odd number.
        :return: Numpy array with rolling mean values calculated over given window length.
        '''
        if window_length % 2 == 0:
            print ("Window length should be an odd number.")
            return
        
        #print(window_length)
        y = np.zeros(len(x))
        for i in range(len(x)):
            if i < window_length/2:
                a = int(i + window_length / 2)
                y[i] = np.mean(x[0:a]) #by row
                
            elif (len(x) - i) < (window_length / 2):
                y[i] = np.mean(x[int((i - window_length / 2)):]) #removes the first (i-window_length/2) elements of x
                
            else:
                a = int(i + window_length / 2)
                b = int(i - window_length / 2)
                y[i] = np.mean(x[b : a])
    
        return y
    
    
    def compute_rolling_std(x, window_length):
        '''
        Method to compute rolling standard deviation.
        :param x: 1D numpy array
        :param window_length: Length of window for computing rolling standard deviation. Must be an odd number.
        :return: Numpy array with rolling standard deviation values calculated over given window length.
        '''
        if window_length % 2 == 0:
            print ("Window length should be an odd number.")
            return
    
        y = np.zeros(len(x))
        for i in range(len(x)):
            if i < window_length/2:
                y[i] = np.std(x[0:int(i + window_length / 2)])
            elif len(x) - i < window_length / 2:
                y[i] = np.std(x[int(i - window_length / 2):])
            else:
                y[i] = np.std(x[int(i - window_length / 2): int(i + window_length / 2)])
    
        return y
    
    
    def detect_hand_movement(raw_accelerometer_data_df, fs, window_length=3, threshold=0.025):
        '''
        Method for detecting hand movement from raw accelerometer data.
        :param raw_accelerometer_data_df: Pandas DataFrame with accelerometer axis represented as x, y and z columns
        :param fs: Sampling rate (samples/second) of the accelerometer data
        :param window_length: Length (in seconds) of the non-overlapping window for hand movement classification
        :param threshold: Threshold value that is applied to the coefficient of variation to detect hand movement
        :return: Detected hand movement as numpy array in desired window length
        '''
        
        #raw_accelerometer_data_df.x=raw_accelerometer_data_df[:,0]
        #raw_accelerometer_data_df.y=raw_accelerometer_data_df[:,1]
        #raw_accelerometer_data_df.z=raw_accelerometer_data_df[:,2]
        
        # Calculate the vector magnitude of the accelerometer signal
        accelerometer_vector_magnitude = np.sqrt(((raw_accelerometer_data_df[:,0])**2 + (raw_accelerometer_data_df[:,1])**2 + (raw_accelerometer_data_df[:,2])**2))
    
        # Low-pass filter the accelerometer vector magnitude signal to remove high frequency components
        low_pass_cutoff = 3 # cutoff frequency for the lowpass filter
        wn = [low_pass_cutoff * 2 / fs]
        [b, a] = signal.iirfilter(6, wn, btype='lowpass', ftype = 'butter')
        accelerometer_vector_magnitude_filt = signal.filtfilt(b, a, accelerometer_vector_magnitude)
    
        # Calculate the rolling coefficient of variation
        rolling_mean = compute_rolling_mean(accelerometer_vector_magnitude_filt, int(fs+1))
        rolling_std = compute_rolling_std(accelerometer_vector_magnitude_filt, int(fs+1))
        rolling_cov = rolling_std/rolling_mean
    
        # Detect CoV values about given movement threshold
        values_above_threshold = (rolling_cov > threshold)*1
    
        # Classify non-overlapping windows as either hand movement or no hand movement
        samples_in_window = window_length * fs
    
        if len(rolling_cov) / samples_in_window > np.floor(len(rolling_cov) / samples_in_window):
            number_of_windows = int(round(len(rolling_cov) / samples_in_window))
        else:
            number_of_windows = int(np.floor(len(rolling_cov) / samples_in_window))
    
        window_labels = np.zeros(number_of_windows)
        for iwin in range(number_of_windows):
    
            if iwin == number_of_windows:
                win_start = iwin * samples_in_window
                win_stop = len(rolling_cov)
            else:
                win_start = iwin * samples_in_window
                win_stop = (iwin + 1) * samples_in_window
    
                if np.mean(values_above_threshold[int(win_start):int(win_stop)]) >= 0.5:
                    window_labels[iwin] = 1
    
        return window_labels, accelerometer_vector_magnitude, accelerometer_vector_magnitude_filt, rolling_cov
    
    
    '''
    Main runner for hand movement detection from accelerometer data located at the wrist location. 
    '''
    
    if ID<12: #filepath of raw accelerometer data
        raw_data_filepath = "C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Converted_csv/SUB%02d_PD.csv" % (ID) # the % starts, the 02 means 2 digits, the d means decimal
    else:
        raw_data_filepath = "C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Converted_csv/SUB%02d_HC.csv" % (ID) 
       
        
    wrist_accel = pd.read_csv(raw_data_filepath)
    wrist_accel = (wrist_accel).values[:,6:9]
    wrist_accel = wrist_accel * (1/8192) #correct units
    
    accel= np.zeros((len(wrist_accel),3))

    file=open("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Mean and std.txt", "r")
    stat=np.array(file.readlines(), dtype="float64")
    
    accel[:,0]= (wrist_accel[:,0]- stat[0])/stat[3] 
    accel[:,1]= (wrist_accel[:,1]- stat[1])/stat[4]
    accel[:,2]= (wrist_accel[:,2]- stat[2])/stat[5]

    sampling_rate = 2000 # Specify sampling rate of sensor data
    window_length = 3 # Specify window length (in seconds) to output hand movement predictions

    # Run hand movement detection
    window_labels, accelerometer_vector_magnitude, accelerometer_vector_magnitude_filt, rolling_cov = detect_hand_movement(accel, sampling_rate)

#%% Save output of detect_hand_movement to two csv files (per subject): one for window_labels, and another for the remaining arrays


    np.savetxt("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Output Hand classifier/window_labels_ID%02d_T0.025.csv" % (ID), window_labels,delimiter=',')
    
    hand_output=np.column_stack((accelerometer_vector_magnitude,accelerometer_vector_magnitude_filt,rolling_cov))
    np.savetxt("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/Output Hand classifier/hand_output_ID%02d_T0.025.csv" % (ID), hand_output ,delimiter=',')



if __name__ == "__main__":
    all_files=init()
    for ID in range(1,17):
       main(ID, all_files)

