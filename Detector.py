# %%
#  "import" de las librerías que vamos a estar usando.
import os
import numpy as np
import mne
from properties import filename
import matplotlib.pyplot as plt                     # Este codigo es un plot basico para ver la señal, los datos concretos.
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from array import array



# versión de MNE, chequear que estamos usando mne3
mne.sys_info()
# Acá leemos un archivo particular
mne.set_log_level("WARNING")
raw= mne.io.read_raw_brainvision('C:\\Users\\julia\\Desktop\\Tesis\\Registros\\RCnew\\ExpS11.vhdr',
    preload=True, 
    eog=('EOG1_1','EOG2_1'),
    misc=('EMG1_1','EMG2_1'),
    verbose=True)
raw.rename_channels(lambda s: s.strip("."))

data =raw.get_data()                                 # Saco los datos concretos, una matriz de numpy
time_shape = data.shape[1]



# -----------------------------------------------------------------
# data(chan,samp), times(1xsamples)

# (1) Aca voy a ver los datos en crudo, ploteandolos por afuera de MNE.  Fijense que los datos estan en Volts lo paso a microvolts.

channel = 0
eeg = raw[channel][0][0][0:250*4]  * pow(10,6)      # Tomo 4 segundos.
print(eeg)

eeg2 = raw[channel][0][0][0:250*1]  * pow(10,6)   #tomo las señales del eeg en el 1 segundo
eeg3 = raw[channel][0][0][250:500*1]  * pow(10,6)   #tomo las señales del eeg en el 2 segundo


dat = np.concatenate( (np.zeros((1,data.shape[1])), data), axis=0) 
print(dat)

i = 0
j=1
while i <= time_shape:
    eeg4 = raw[channel][0][0][i:250*j*1]  * pow(10,6) # tomo los valores del eeg cada 1 segundo
    print(eeg4)
  
    signal=eeg4 - np.mean(eeg4) # le resto la media de la señal a mi señal 
    signal=signal+np.min(signal)*-1 # le sumo a la señal el minimo en positivo
    #print(signal)
    
    newsignal=signal

    newsignal[signal>75]=100
    newsignal[signal<=75]=0

    #print(newsignal)
    
    #print(i)
    #print(j)

    i += 250
    j += 1

    

