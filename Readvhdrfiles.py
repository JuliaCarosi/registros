# %%
#  "import" de las librerías que vamos a estar usando.
import os
import numpy as np
import mne

from matplotlib.transforms import Bbox

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def savesubfigure(a,filename):
    allaxes = a.get_axes()
    subfig = allaxes[0]

    # Save just the portion _inside_ the second axis's boundaries
    extent = full_extent(subfig).transformed(a.dpi_scale_trans.inverted())

    for item in ([subfig.title, subfig.xaxis.label, subfig.yaxis.label] +
                 subfig.get_xticklabels() + subfig.get_yticklabels()):
        item.set_fontsize(14)

    # Alternatively,
    # extent = ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    a.savefig(filename, bbox_inches=extent)

# versión de MNE, chequear que estamos usando mne3
mne.sys_info()
# Acá leemos un archivo particular
mne.set_log_level("WARNING")
raw= mne.io.read_raw_brainvision('/Users/rramele/work/kcomplexes/drive-download-20200409T185153Z-001/ExpS11.vhdr',
#raw= mne.io.read_raw_brainvision('C:\\Users\\julia\\Desktop\\Tesis\\Registros\\RCnew\\ExpS11.vhdr', 
    preload=True, 
    eog=('EOG1_1','EOG2_1'),
    misc=('EMG1_1','EMG2_1'),
    verbose=True)
raw.rename_channels(lambda s: s.strip("."))
#raw.set_montage("standard_1020")
#raw.set_eeg_reference("average")

# data(chan,samp), times(1xsamples)
a = raw[0]

channel = 0 
# Channel data
print(raw[channel][0].shape)

raw[channel][0][0][100] = 10000

#eeg_mne.filter(1,20)

raw.plot_psd()
#raw.filter(1,20)
#raw.plot_psd()

#a = raw.plot(show_options=True,title='KComplex',start=2121,duration=30,n_channels=10, scalings='auto', show_first_samp=True)
#savesubfigure(a,'kcomplex1.eps')

eeg = raw[channel][0][0][0:250*4]  * pow(10,6)

raw[channel][0][0][0:250*4] = np.ones((1,250*4))

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(eeg,'r', label='EEG')
plt.legend(loc='upper left')
plt.show(block=False)

#a = raw.plot(show_options=True,title='KComplex2',start=504,duration=30,n_channels=10, scalings=dict(eeg=20e-6))
# chequean que la frecuencia de sampleo sea la esperada
print('Sampling Frequency: %d' %  raw.info['sfreq'] )

pplot = raw.plot(scalings='auto',n_channels=10,block=True, )

allaxes = pplot.get_axes()
subfig = allaxes[0]
subfig.figure.canvas.draw()

print(raw.annotations)

print (raw.annotations.description)

raw.annotations.save("Annotations.txt")

# %%
