import os.path
import random

import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io.wavfile import write

C = 340
fs = 16e3
reciver_position = [2, 1, 1]
room_dimentions = [4, 5, 3]
reverberation_time = 0.4
nsample = int(reverberation_time * fs)
R = 2
DATASET_SIZE = 10000
DATASET_PATH = r'C:\Users\reiem\PycharmProjects\Acoustic_Locating_VQ-VAE\rir_dataset_generator\rir_dataset'

if (room_dimentions[0] - reciver_position[0]) > R:
    max_theta = np.pi / 2
else:
    max_theta = np.arcsin((room_dimentions[0] - reciver_position[0]) / R)

if (reciver_position[0] - R) > 0:
    min_theta = -np.pi / 2
else:
    min_theta = np.arcsin(-reciver_position[0] / R)

theta_samples = 5
theta_pos = np.linspace(min_theta, max_theta, theta_samples)

if R > room_dimentions[2]:
    raise ValueError('R is greater than room_dimentions length')

positions_x, positions_y = (reciver_position[0] + R * np.sin(theta_pos), reciver_position[1] + R * np.cos(theta_pos))

fix, ax = plt.subplots(1, 1)
plt.scatter(positions_x, positions_y, label='Source locations')
plt.scatter(reciver_position[0], reciver_position[1], label='Mic location')
ax.add_patch(Rectangle((0, 0), room_dimentions[0], room_dimentions[1], alpha=0.1, color='red'))
ax.set_xlabel('Room X')
ax.set_ylabel('Room Y')
ax.legend()
plt.show()


H = []

for i, (i_x, i_y) in enumerate(zip(positions_x, positions_y)):
    print(f'{i}Generating H for location {i_x, i_y}')

    h = rir.generate(
        c=C,  # Sound velocity (m/s)
        fs=fs,  # Sample frequency (samples/s)
        r=reciver_position,  # Receiver position(s) [x y z] (m)
        s=[i_x, i_y, reciver_position[2]],  # Source position [x y z] (m)
        L=room_dimentions,  # Room dimensions [x y z] (m)
        reverberation_time=reverberation_time,  # Reverberation time (s)
        nsample=nsample,  # Number of output samples
    )
    H.append(h)

res = list(zip(H, positions_x, positions_y, np.ones(len(positions_x), dtype=float)))
for j in range(DATASET_SIZE):
    print(f'Generating {j}')
    h, i_x, i_y, i_z = random.sample(res, k=1)[0]
    noise = np.random.normal(0, 1, size=nsample*7)
    signal = ss.convolve(h, np.expand_dims(noise, 1))

    filename = f'rir_R_{room_dimentions[0]}_{room_dimentions[1]}_{room_dimentions[2]}_M_{reciver_position[0]}_{reciver_position[1]}_{reciver_position[2]}_S_{i_x:.1f}_{i_y:.1f}_{i_z:.1f}_{j}.wav'
    scaled = np.int16(signal / np.max(np.abs(signal)) * 32767)
    write(os.path.join(DATASET_PATH,filename), int(fs), scaled)

# print(signal.shape)         # (11462, 2)
#
# # Convolve 2-channel signal with 3 impulse responses

#
# print(signal.shape)         # (15557, 2, 3)
