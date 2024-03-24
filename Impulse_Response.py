import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import matplotlib.pyplot as plt

import os
import urllib.request
import tarfile

from scipy import signal


data_dir = 'data/'
data_url = 'https://www.openslr.org/resources/12/dev-clean.tar.gz'
filename = data_url.rsplit('/', 1)[-1]
filepath = os.path.join(data_dir, filename)
if not os.path.isfile(filepath):
    print('Downloading data')
    urllib.request.urlretrieve(data_url, filepath)
    print('Finished downloading, unpacking data')
    downloaded = tarfile.open(filepath)
    downloaded.extractall(data_dir)
    downloaded.close()

orig_signal, fs = sf.read("data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac", always_2d=True)
f, t, Zxx = signal.stft(np.squeeze(orig_signal), fs)

h = rir.generate(
    c=340,                  # Sound velocity (m/s)
    fs=fs,                  # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [2, 1.5, 1]
    ],
    s=[2, 3.5, 2],          # Source position [x y z] (m)
    L=[5, 4, 6],            # Room dimensions [x y z] (m)
    reverberation_time=0.4, # Reverberation time (s)
    nsample=4096,           # Number of output samples
)

print(h.shape)              # (4096, 3)
print(orig_signal.shape)         # (11462, 2)

# Convolve 2-channel signal with 3 impulse responses
signal = ss.convolve(h[:, None, :], orig_signal[:, :, None])

print(signal.shape)         # (15557, 2, 3)

fig, axes = plt.subplots(3)
axes[0].plot(orig_signal)
axes[0].title.set_text('Input signal')

axes[1].plot(h)
axes[1].title.set_text('Room Impulse Response')
axes[2].plot(signal[:, 0, :])
axes[2].title.set_text('Output signal')

plt.tight_layout()  # Fix subplot spacing
plt.show()

amp = 2 * np.sqrt(2)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=np.abs(Zxx).min(), vmax=np.abs(Zxx).max())
plt.title('STFT Magnitude, original signal')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()