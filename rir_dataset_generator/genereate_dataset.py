import os.path
import random

import numpy as np
import scipy.signal as ss
import rir_generator as rir
import matplotlib.pyplot as plt
import torchaudio
from matplotlib.patches import Rectangle
from scipy.io.wavfile import write
from torch.utils.data import DataLoader


def get_h_list(positions_x: np.ndarray, positions_y: np.ndarray, C: int, fs: float, reciver_position: list,
               reverberation_time: float, nsample: int) -> list:
    H = []

    for i, (i_x, i_y) in enumerate(zip(positions_x, positions_y)):
        print(f'{i}Generating H for location {i_x, i_y}')

        h = rir.generate(
            c=C,  # Sound velocity (m/s)
            fs=fs,  # Sample frequency (samples/s)
            r=reciver_position,  # Receiver position(s) [x y z] (m)
            s=[i_x, i_y, reciver_position[2]],  # Source position [x y z] (m)
            L=room_dimensions,  # Room dimensions [x y z] (m)
            reverberation_time=reverberation_time,  # Reverberation time (s)
            nsample=nsample,  # Number of output samples
        )
        H.append(h)
    return H


def get_positions_for_ray(room_dimensions: list, receiver_position: list, theta_samples=5) -> tuple[
    np.ndarray, np.ndarray]:
    if (room_dimensions[0] - receiver_position[0]) > R:
        max_theta = np.pi / 2
    else:
        max_theta = np.arcsin((room_dimensions[0] - receiver_position[0]) / R)

    if (receiver_position[0] - R) > 0:
        min_theta = -np.pi / 2
    else:
        min_theta = np.arcsin(-receiver_position[0] / R)

    theta_pos = np.linspace(min_theta, max_theta, theta_samples)

    if R > room_dimensions[2]:
        raise ValueError('R is greater than room_dimentions length')

    positions_x, positions_y = (
        receiver_position[0] + R * np.sin(theta_pos), receiver_position[1] + R * np.cos(theta_pos))
    return positions_x, positions_y


def plot_positions_in_room(positions_x, positions_y, reciver_position, room_dimentions):
    fix, ax = plt.subplots(1, 1)
    plt.scatter(positions_x, positions_y, label='Source locations')
    plt.scatter(reciver_position[0], reciver_position[1], label='Mic location')
    ax.add_patch(Rectangle((0, 0), room_dimentions[0], room_dimentions[1], alpha=0.1, color='red'))
    ax.set_xlabel('Room X')
    ax.set_ylabel('Room Y')
    ax.legend()
    plt.show()


def write_dataset_to_disk(H: list, positions_x, positions_y, reciver_position, room_dimentions, speach_loader):
    res = list(zip(H, positions_x, positions_y, np.ones(len(positions_x), dtype=float)))
    for j in range(DATASET_SIZE):
        print(f'Generating {j}')
        h, i_x, i_y, i_z = random.sample(res, k=1)[0]
        # noise = np.random.normal(0, 1, size=nsample * 7)
        (waveform, sample_rate, _, _, _, _) = next(iter(speach_loader))
        waveform = waveform.view(-1,1)
        signal = (ss.convolve(waveform,h,mode='same')/waveform).numpy()

        filename = f'rir_R_{room_dimentions[0]}_{room_dimentions[1]}_{room_dimentions[2]}_M_{reciver_position[0]}_{reciver_position[1]}_{reciver_position[2]}_S_{i_x:.1f}_{i_y:.1f}_{i_z:.1f}_{j}.wav'
        scaled = np.int16(signal / np.max(np.abs(signal)) * 32767)
        write(os.path.join(DATASET_PATH, filename), int(fs), scaled)


if __name__ == '__main__':

    def data_preprocessing(data):
        SAMPLING_RATE = 16e3
        NFFT = int(SAMPLING_RATE * 0.025)
        HOP_LENGTH = int(SAMPLING_RATE * 0.01)
        audio_transformer = torchaudio.transforms.Spectrogram(n_fft=NFFT, hop_length=HOP_LENGTH, power=1, center=True,
                                                              pad=0, normalized=True)
        spectrograms = []
        for (waveform, sample_rate, _, _, _, _) in data:
            spec = audio_transformer(waveform)
            spectrograms.append(spec)

        return spectrograms, sample_rate,

    C = 340
    fs = 16e3
    receiver_position = [2, 1, 1]
    room_dimensions = [4, 5, 3]
    reverberation_time = 0.4
    nsample = int(reverberation_time * fs)
    R = 2
    DATASET_SIZE = 5000
    DATASET_PATH = r'C:\Users\reiem\PycharmProjects\Acoustic_Locating_VQ-VAE\rir_dataset_generator\rir_dataset'
    SPEACH_PATH = r'C:\Users\reiem\PycharmProjects\Acoustic_Locating_VQ-VAE\BasedOnVQ-VAE-Speech\data'
    speach_dataset = torchaudio.datasets.LIBRISPEECH(SPEACH_PATH, url='train-clean-100', download=True)
    speach_loader = DataLoader(speach_dataset, batch_size=1, shuffle=True)

    positions_x, positions_y = get_positions_for_ray(room_dimensions, receiver_position)
    plot_positions_in_room(positions_x, positions_y, receiver_position, room_dimensions)
    H = get_h_list(positions_x, positions_y, C, fs, receiver_position, reverberation_time, nsample)
    write_dataset_to_disk(H, positions_x, positions_y, receiver_position, room_dimensions, speach_loader)
