import torch
import librosa
from acoustic_locating_vq_vae.data_preprocessing import batchify_spectrograms
import os
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

from acoustic_locating_vq_vae.rir_dataset_generator.speech_dataset import speech_DATASET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.load(os.path.join(os.getcwd(), r'C:\Users\reiem\PycharmProjects\Acoustic_Locating_VQ-VAE\models\model_speech.pt'))

fs = 16e3
dataset_path = r"C:\Users\reiem\PycharmProjects\Acoustic_Locating_VQ-VAE\scripts\speech_dataset\dev_data"
batch_size = 1
NFFT = 2**11
olap = 0.75
noverlap = round(olap * NFFT)

def sound_from_sample(data, fs, filename):
    fs = int(fs)
    # S = librosa.feature.inverse.mel_to_stft(data, n_fft=int(fs * 0.025), power=1.0)
    y = librosa.griffinlim(data)
    scaled = np.int16(y / np.max(np.abs(y)) * 32767).T
    write(filename, fs, scaled)

test_data = speech_DATASET(dataset_path)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                         collate_fn=lambda x: batchify_spectrograms(x, NFFT, noverlap))

model.eval()

(originals,_) = next(iter(test_loader))
originals = torch.abs(originals)
originals_db = librosa.power_to_db(originals, ref=np.max)
originals = originals.to(device)
originals = torch.squeeze(originals, dim=1)
originals.permute(0,2,1)
_, reconstructions, _ = model(originals)
reconstructions_db = librosa.power_to_db(reconstructions.detach().cpu(), ref=np.max)


fig, axes = plt.subplots(2)

img = librosa.display.specshow(originals_db.squeeze(), x_axis='time',
                               y_axis='mel', sr=fs,
                               fmax=8000, ax=axes[0])

fig.colorbar(img, ax=axes[0], format='%+2.0f dB')
axes[0].set(title='Mel-frequency spectrogram')

img2 = librosa.display.specshow(reconstructions_db.squeeze(), x_axis='time',
                               y_axis='mel', sr=fs,
                               fmax=8000, ax=axes[1])

fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
axes[1].set(title='Mel-frequency spectrogram')


plt.show()

sound_from_sample(originals.cpu().numpy(), fs, 'test.wav')
sound_from_sample(reconstructions.cpu().detach().numpy(), fs, 'recon.wav')