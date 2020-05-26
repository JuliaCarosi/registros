# %%
#  "import" de las librerías que vamos a estar usando.
import numpy as np
import mne
import os
import matplotlib.pyplot as plt                     # Este codigo es un plot basico para ver la señal, los datos concretos.
import argparse
import easygui

from tkinter import messagebox
#from properties import filename
from scipy import signal
from matplotlib.transforms import Bbox

#Print the system information
mne.sys_info()

def get_name(path):
    file = path.split(os.path.sep)[-1]
    print('file', file)
    name = file.split('/')[-1]
    print('name: ',name)
    subject = name.split('.')[0]
    print('subject',subject)
    return subject


def set_sleep_states(raw,path):
    estados = np.loadtxt(path,delimiter =' ', usecols =(0) )
    cant_anotations= estados.shape[0]
 
    onset = np.zeros((cant_anotations))        #array tamaño 136=4080/30 donde se van a mostrar las anotaciones
    duration = np.zeros((cant_anotations))     #array duracion, siempre 30
    description = np.zeros((cant_anotations))  #array descripcion de la etiqueta
    start = 0
    for i in range(cant_anotations):
        start= start + 30 #* sfreq
        onset[i] = start
        duration[i] = 30 #* sfreq
        description[i] = estados[i]

    states_anotations = mne.Annotations(onset,duration,description)    #<Annotations | 136 segments: 0.0 (50), 1.0 (20), 2.0 (42), 3.0 (24)>
    raw2 = raw.copy().set_annotations(states_anotations)
    
    return raw2

def extract_signal(raw):
    for element in raw.annotations:
        if (element.description == 'K'):
            print('onset: ', element.get('onset'), 'duration: ', element.get('duration'))


def main():
    scal = dict(mag=1e-12, grad=4e-11, eeg=20e-5, eog=150e-6, ecg=5e-4,
            emg=1e-4, ref_meg=1e-12, misc=1e-3, stim=1,
            resp=1, chpi=1e-4, whitened=1e2)

    messagebox.showinfo(message="Selecciona el archivo vhdr", title="Seleccion de datos")
    path = easygui.fileopenbox(title='Seleccione vhdr')#selecciono la carpeta vhdr
    subject = get_name(path)
    
    mne.set_log_level("WARNING")
    raw= mne.io.read_raw_brainvision(path, 
        preload=True, 
        eog=('EOG1_1','EOG2_1'),
        misc=('EMG1_1','EMG2_1'),
        verbose=True)
    raw.rename_channels(lambda s: s.strip("."))

    info = raw.info
    sfreq = info.get('sfreq') #frecuencia de muestreo
    
    anotaciones = messagebox.askquestion(message="El archivo ya posee anotaciones?", title="Anotaciones")
    if (anotaciones == 'yes'):
        messagebox.showinfo(message="Selecciona txt con etapas de sueño", title="Seleccion de Etapas de sueño")
        path_annotations = easygui.fileopenbox(title='Seleccione txt con anotaciones') #selecciono el txt de anotaciones anteriores
        raw.set_annotations(mne.read_annotations(path_annotations))
    elif(anotaciones == 'no'):
        messagebox.showinfo(message="Selecciona el txt con anotaciones", title="Seleccion de anotaciones")
        path_states = easygui.fileopenbox(title='Seleccione txt con etapa de sueño') #selecciono el txt de estados de sueño
        raw = set_sleep_states(raw,path_states)

    # -----------------------------------------------------------------
    data =raw.get_data()         # data(chan,samp), times(1xsamples)
    info = raw.info              #info
    sfreq = info.get('sfreq')    #frecuencia de muestreo

    #Con este código extraigo los datos que necesito y me rearmo la estructura que necesito para poder analizarlo mejor
    data =raw.get_data()                                 # Saco los datos concretos, una matriz de numpy
    new_data=data.copy()
    canal_eogs = data[6,:] - data[7,:]                   # Cree la variable de la resta de las dos señales
    canal_emgs =  data[8,:] - data[9,:]
    t = np.linspace(1, round(data.shape[1]/sfreq), data.shape[1], endpoint=False)   
    new_data[0]= signal.square(2 * np.pi * 1 * t)
    new_data[1]=data[[0], :]
    new_data[2]=data[[1], :]
    new_data[3]=canal_eogs
    new_data[4]=canal_emgs

    new_data=new_data[[0,1,2,3,4], :]        #Elimino los otros canales

    new_chnames =[ "Pulso", raw.ch_names[0], raw.ch_names[1], "EOG_resta", "EMG_resta"] 
    new_chtypes = ['misc'] +['eeg' for _ in new_chnames[0:2]] + ['misc','misc'] # Recompongo los canales.

    new_info = mne.create_info(new_chnames, sfreq, ch_types=new_chtypes)
    new_info['meas_date'] = raw.info['meas_date']       # Registro el timestamp para las anotaciones.

    new_raw=mne.io.RawArray(new_data, new_info)
    new_raw.set_annotations(raw.annotations)           # Construyo un nuevo objeto raw que tiene lo que necesito.

    scal = dict(mag=1e-12, grad=4e-11, eeg=20e-5, eog=150e-6, ecg=5e-4,emg=1e-4, ref_meg=1e-12, misc=1e-3, stim=1,
        resp=1, chpi=1e-4, whitened=1e2)

    #pplot=new_raw.plot(scalings=scal, duration=30, n_channels=10, block=True, )
        
    new_raw.plot(show_options=True,title='Etiquetado',start=0,duration=30,n_channels=10, scalings=scal,block=True)
    new_raw.annotations.save(subject + "Annotations.txt")



if __name__ == '__main__':
    #se agregan todos los parametros que pueden pasarse al software cuando se llama
    parser = argparse.ArgumentParser()
    #parser.add_argument('--anotaciones', required=True, help='True o False si el archivo ya tiene anotaciones previas')
    # parser.add_argument('--posee_annotations', required=True, default=False, help='Ingresar True si ya tiene anotaciones guardadas')
    args = parser.parse_args()

    main(**vars(args))
