import numpy as np
import mne
import os
import matplotlib.pyplot as plt                   .
import argparse
import easygui

from datetime import datetime    
from tkinter import messagebox
from scipy import signal


def get_name(path):
    file = path.split(os.path.sep)[-1]
    name = file.split('/')[-1]
    subject = name.split('.')[0]
    return subject


def set_sleep_states(raw,path):
    estados = np.loadtxt(path,delimiter =' ', usecols =(0) )
    cant_anotations= estados.shape[0]
 
    onset = np.zeros((cant_anotations))        #array tamaño 136=4080/30 donde se van a mostrar las anotaciones
    duration = np.zeros((cant_anotations))     #array duracion, siempre 30
    description = np.zeros((cant_anotations))  #array descripcion de la etiqueta
    start = 0
    for i in range(cant_anotations):
        onset[i] = start
        duration[i] = 30 #* sfreq
        description[i] = estados[i]
        start= start + 30 #* sfreq

    states_anotations = mne.Annotations(onset,duration,description, orig_time=raw.annotations.orig_time)    #<Annotations | 136 segments: 0.0 (50), 1.0 (20), 2.0 (42), 3.0 (24)>
    raw2 = raw.copy().set_annotations(states_anotations)

    return raw2, states_anotations


def new_raw_data(raw,sfreq):
    data =raw.get_data()                    # Saco los datos concretos, una matriz de numpy
    time_shape = data.shape[1]
    
    # Con este código extraigo los datos que necesito y me rearmo la estructura 
    eog = (raw.copy()).pick_types(eog=True)
    eog_data = eog.get_data()
    sub_eog = eog_data[0,:]-eog_data[1,:]

    emg = (raw.copy()).pick_types(misc=True)
    emg_data = emg.get_data()
    sub_emg = emg_data[0,:]-emg_data[1,:]

    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    #me creo mi señal artificial con pulso de 0.5 seg
    pulso = signal.square(2 * np.pi * 1 * t) #señal del pulso

    pos = (raw.ch_names).index('C3_1')
    c3_1 = data[pos,:]

    pos2 =(raw.ch_names).index('C4_1')
    c4_1 = data[pos2,:]

    new_data=data.copy()
    new_data[0]= sub_eog
    new_data[1]= sub_emg
    new_data[2]= pulso
    new_data[3]= c3_1
    new_data[4]= c4_1
    new_data=new_data[[0,1,2,3,4], :]
    new_ch_names = ['EOG', 'EMG', 'Pulse', 'C3', 'C4']
    order=[0,3,2,4,1]   
    new_chtypes = 3* ['misc'] + 2 *['eeg'] # Recompongo los canales.
    
    new_info = mne.create_info(new_ch_names, sfreq, ch_types=new_chtypes)
    new_info['meas_date'] = raw.info['meas_date']       # Registro el timestamp para las anotaciones.

    new_raw=mne.io.RawArray(new_data, new_info)
    new_raw.set_annotations(raw.annotations)           # Construyo un nuevo objeto raw que tiene lo que necesito.

    return(new_raw)


def main():
    scal = dict(mag=1e-12, grad=4e-11, eeg=20e-5, eog=150e-6, ecg=5e-4,
            emg=1e-4, ref_meg=1e-12, misc=1e-3, stim=1,
            resp=1, chpi=1e-4, whitened=1e2)
    
    anotaciones = messagebox.askquestion(message="¿El archivo ya posee anotaciones?", title="Anotaciones")
    if (anotaciones == 'no'):
        messagebox.showinfo(message="Selecciona el archivo vhdr", title="Seleccion de datos")
        path = easygui.fileopenbox(title='Seleccione vhdr')#selecciono la carpeta vhdr

        mne.set_log_level("WARNING")
        raw= mne.io.read_raw_brainvision(path, 
            preload=True, 
            eog=('EOG1_1','EOG2_1'),
            misc=('EMG1_1','EMG2_1'),
            verbose=True)
        raw.rename_channels(lambda s: s.strip("."))
    
        messagebox.showinfo(message="Selecciona txt con etapas de sueño", title="Seleccion de Etapas de sueño")
        path_states = easygui.fileopenbox(title='Seleccione txt con las etapas de sueño') #selecciono el txt de anotaciones anteriores
        raw,_ = set_sleep_states(raw,path_states)

        info = raw.info
        sfreq = info.get('sfreq') #frecuencia de muestreo
        raw = new_raw_data(raw,sfreq)  #re-armo la estructura

    elif(anotaciones == 'yes'): #evita pasar por todo lo mismo de nuevo 
        messagebox.showinfo(message="Selecciona el archivo fif", title="Seleccion de datos")
        path = easygui.fileopenbox(title='Seleccione fif') #selecciono la carpeta vhdr
        raw = mne.io.read_raw_fif(path) 

        messagebox.showinfo(message="Selecciona txt con etapas de sueño", title="Seleccion de Etapas de sueño")
        path_states = easygui.fileopenbox(title='Seleccione txt con las etapas de sueño') #selecciono el txt de anotaciones anteriores
        _,annot_sleep= set_sleep_states(raw,path_states)

        annot=raw.annotations
        raw.set_annotations(annot + annot_sleep)

    info = raw.info
    sfreq = info.get('sfreq')                      #frecuencia de muestreo


    subject = get_name(path)                       #obtengo el nombre del sujeto del archivo

    #si ya etiquete, elimino la fecha 
    if '2020_' in subject:
        ind = str.index(subject,'2020')
        subject=subject[:ind]

    #se guarda como sujeto+fecha
    d=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    s=subject+d
    
    raw.plot(show_options=True,title='Etiquetado',start=0,duration=30,n_channels=5, scalings=scal,block=True,order=[0,3,2,4,1])
    #order cambia el orden en el que aparecen los canales

    raw.annotations.save(s+ "Annotations.txt")
    raw.save(s +  ".fif",overwrite=True)
    
    
main()
