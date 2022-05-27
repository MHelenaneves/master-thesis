# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:56:28 2022

@author: mhele
"""

#%%

all_files.to_pickle("all_files.pkl")

#%%
a_dataframe = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

a_dataframe.to_pickle("a_file.pkl")

output = pd.read_pickle("a_file.pkl")


#%%
pd.concat(all_files).to_csv('all_files')  
    
#%%
def filtering_emg_cheby(emg):
    emg_correctmean = emg - np.mean(emg)
    #N=8
    #sos = signal.ellip(N, Ap, As, wc, 'bandpass', output='sos')
    #y_sos = signal.sosfilt(sos, emg)

    b,a = signal.cheby1(2, 1, [2,10], 'bandpass',fs=2000)
    #y_sos = signal.sosfilt(b,a, emg_correctmean)
    
    
    emg_filtered = sp.signal.filtfilt(b, a, emg_correctmean) #bandpass
    emg_normalized_cheby= (emg_filtered- np.mean(emg_filtered))/np.std(emg_filtered)

#    plt.figure()
#    plt.plot(emg_norm)
#    plt.title('Signal after Chebyshev ')
#    plt.xlabel('Frequency [Hz]')
#    plt.ylabel('EMG (V)')
    
    
    return emg_normalized_cheby

emg_normalized_cheby=filtering_emg_cheby(emg)


#%%
plt.figure()
f, t, Sxx = signal.spectrogram(emg, fs)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
plt.show()

# plt.figure()
# plt.specgram(emg,fs)

# plt.figure()
# plt.specgram(emg_normalized,fs)

#%%
#plt.savefig('PD_ID8_EMG_Filtered3-60.png')


#%% Filtering in the rest periods

#indexes 2 and 3 for ID10
beg=int(indexes[1])
#end=int(indexes[2])

#plt.figure()
#plt.plot(emg[beg:end])



#%%
start=int(indexes[0])
plt.figure()
for i in range(0,len(indexes)) :
   x=indexes[i]
   plt.axvline(x, color="blue")
   
   
plt.plot(emg_normalized[start:-1], 'r-')
plt.show()



#%%
freqs, times, spectrogram = signal.spectrogram(emg)

plt.figure()
plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')


#%%
f, Pxx_den = signal.periodogram(emg, fs)
plt.semilogy(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

#%%

plt.figure()
plt.plot(np.fft.rfft(emg))


#%%
    #plt.figure()
    #plt.magnitude_spectrum(emg_normalized, fs, scale='dB', color='C1')


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

#%% First attempt at filtering
emg, emg_correctmean,emg_filtered,emg_normalized=filtering_emg(emg)