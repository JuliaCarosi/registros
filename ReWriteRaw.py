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

# -----------------------------------------------------------------
# data(chan,samp), times(1xsamples)

# (1) Aca voy a ver los datos en crudo, ploteandolos por afuera de MNE.  Fijense que los datos estan en Volts lo paso a microvolts.

channel = 0
eeg = raw[channel][0][0][0:250*4]  * pow(10,6)      # Tomo 4 segundos.


fig = plt.figure()
ax1 = fig.add_subplot(111)

#ax1.plot(eeg,'r', label='EEG')
plt.legend(loc='upper left')
plt.show(block=False)

#a = raw.plot(show_options=True,title='KComplex2',start=504,duration=30,n_channels=10, scalings=dict(eeg=20e-6))
# chequean que la frecuencia de sampleo sea la esperada
print('Sampling Frequency: %d' %  raw.info['sfreq'] )

sfreq = raw.info['sfreq']
data =raw.get_data()                            # Saco los datos concretos, una matriz de numpy
t = np.linspace(1, round(data.shape[1]/sfreq), data.shape[1], endpoint=False)   
print(data[0:5])
canal_eogs = data[6,:] - data[7,:]                   # Cree la variable de la resta de las dos señales
canal_emgs =  data[8,:] - data[9,:]
data[6]=canal_eogs
data[7]=canal_emgs
data[8]=signal.square(2 * np.pi * 1 * t)
data=data[[0,1,2,3,4,5,6,7,8], :]
#pplot = raw.plot(scalings='auto',n_channels=10,block=True, )
#t = np.linspace(0, 3318, 663804, endpoint=False)    #me creo mi señal con pulso de 0.5 seg
plt.plot(t, signal.square(2 * np.pi * 1 * t))
plt.ylim(-2, 2)

# (2) Con este código extraigo los datos que necesito y me rearmo la estructura que necesito para poder analizarlo mejor
new_ch_names =[ raw.ch_names[0], raw.ch_names[1],raw.ch_names[2] , raw.ch_names[3],  raw.ch_names[4],  raw.ch_names[5], "EOG_resta", "EMG_resta", "Pulso"] 


ch_names = ['Supera75'] + new_ch_names              # Saco el nombre de los canales pero agrego uno 'peak'


dat = np.concatenate( (np.zeros((1,data.shape[1])), data), axis=0)    # Le agrego a los datos un array con zeros.

                                                                # aca si ustedes quieren pueden agregarle 20 o algo asi
                                                                      # cada vez que la señal que ustedes analizan supera los 75




dat = np.concatenate( (np.zeros ((1, canal_eogs.shape[0])), data), axis=0) 
dat = np.concatenate( (np.zeros ((1, canal_emgs.shape[0])), data), axis=0) 

ch_types = ['misc'] + ['eeg' for _ in ch_names[0:6]] + ['misc','misc'] + ['misc']  # Recompongo los canales.
info = mne.create_info(ch_names, sfreq, ch_types=ch_types)



info['meas_date'] = raw.info['meas_date']       # Registro el timestamp para las anotaciones.

reraw = mne.io.RawArray(dat, info)
reraw.set_annotations(raw.annotations)          # Construyo un nuevo objeto raw que tiene lo que necesito.

#reraw.plot(scalings='auto',n_channels=10,block=True, )
#pplot=reraw.plot(scalings='auto', n_channels=10, block=True, )

scal = dict( eog=250e-6,emg=1e-4, grad=4e-11,  eeg=20e-5)     #Scaling factors 

reraw_copy=reraw.copy()
reraw_copy.drop_channels(['F3_1','F4_1','P3_1','P4_1'])
pplot=reraw_copy.plot(scalings='auto', duration=30, n_channels=10, block=True, )

# POR Ejemplo con esto pueden restar EMG1 y EMG2 y dejar solo uno, lo mismo con EOG.
