import os.path
import git
import random

import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io.wavfile import write
import torchaudio
import torch
from torch.utils.data import DataLoader
from pathlib import Path


def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


C = 340
fs = 16e3
receiver_position = [2, 1.5, 1.5]
room_dimensions = [4, 5, 3]
reverberation_time = 0.4
n_sample = int(reverberation_time * fs)
R = 1
DATASET_SIZE = 100

git_root_path = get_git_root(os.getcwd())
LibriSpeech_PATH = os.path.join(git_root_path, 'data')
DATASET_DEST_PATH = os.path.join(git_root_path, 'rir_dataset_generator', 'dev_data')
Path(DATASET_DEST_PATH).mkdir(parents=True, exist_ok=True)

NFFT = int(fs * 0.025)
HOP_LENGTH = int(fs * 0.01)

audio_transformer = torchaudio.transforms.Spectrogram(n_fft=NFFT, hop_length=HOP_LENGTH, power=1,
                                                      center=True, pad=0, normalized=True)


def data_preprocessing(data):
    theta = np.random.uniform(low=-np.pi, high=np.pi, size=1)
    z_loc = np.random.uniform(low=0, high=1, size=1)
    h_src_loc = np.stack((R*np.cos(theta).T, R*np.sin(theta).T, z_loc.T), axis=1) + receiver_position
    h_src_loc = np.minimum(h_src_loc, room_dimensions)
    h_RIR = rir.generate(
        c=C,  # Sound velocity (m/s)
        fs=int(fs),  # Sample frequency (samples/s)
        r=receiver_position,
        s=np.squeeze(h_src_loc),  # Source position [x y z] (m)
        L=room_dimensions,  # Room dimensions [x y z] (m)
        reverberation_time=reverberation_time,  # Reverberation time (s)
        nsample=n_sample,  # Number of output samples
    )

    for (waveform, sample_rate, _, _, _, _) in data:
        spec_signal = audio_transformer(waveform)
        waveform_h = ss.convolve(waveform.squeeze(), h_RIR.squeeze(), mode='same')
        spec_with_h = audio_transformer(torch.from_numpy(waveform_h))

        spec = np.divide(spec_signal, spec_with_h)
        spec_final = np.divide(spec, np.abs(spec).max())

    return spec_final, sample_rate, theta  # transcript, speaker_id, chapter_id, utterance_id


train = torchaudio.datasets.LIBRISPEECH(LibriSpeech_PATH, url='train-clean-100', download=True)
train_loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=lambda x: data_preprocessing(x))

theta_array = []

for i_sample in range(DATASET_SIZE):
    print('Generating sample: ', i_sample)
    (spec_final, sample_rate, theta) = next(iter(train_loader))
    filename = os.path.join(DATASET_DEST_PATH, f'{i_sample}.pt')
    theta_array.append(theta)
    scaled = np.int16(spec_final / np.abs(spec_final).max() * 32767)
    torch.save(scaled, filename)

torch.save(theta_array, os.path.join(DATASET_DEST_PATH, 'theta.pt'))
