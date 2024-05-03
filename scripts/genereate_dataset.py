import os.path

import numpy as np
import scipy.signal as ss
import rir_generator as rir
import torchaudio
import torch
from torch.utils.data import DataLoader
from pathlib import Path


def data_preprocessing(data, Z_LOC_SOURCE,R, room_dimensions,receiver_position,fs,reverberation_time, n_sample, audio_transformer,C, **kwargs ):
    theta = np.random.uniform(low=-np.pi, high=np.pi, size=1)
    z_loc = np.array([Z_LOC_SOURCE])
    h_src_loc = np.stack((R * np.cos(theta).T, R * np.sin(theta).T, z_loc), axis=1) + receiver_position
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
        spec_signal = np.squeeze(audio_transformer(waveform))
        waveform_h = ss.convolve(waveform.squeeze(), h_RIR.squeeze(), mode='same')
        spec_with_h = audio_transformer(torch.from_numpy(waveform_h))

        spec = np.divide(spec_signal, spec_with_h + 1e-8)
        spec_final = np.divide(spec, np.abs(spec).max())

        winner_est = torch.sum(spec_with_h * np.conjugate(spec_signal), dim=1) / (
                    torch.sum(spec_signal * np.conjugate(spec_signal), dim=1) + 1e-8)

    return spec_final, sample_rate, theta, winner_est  # transcript, speaker_id, chapter_id, utterance_id


def get_dataset_params(data_type: str) -> dict:
    params = {}
    params['fs'] = int(16e3)
    params['NFFT'] = int(params['fs'] * 0.025)
    params['HOP_LENGTH'] = int(params['fs'] * 0.01)

    if data_type == 'rir':
        params['C'] = 340
        params['receiver_position'] = [2, 1.5, 1.5]
        params['room_dimensions'] = [4, 5, 3]
        params['reverberation_time'] = 0.4
        params['n_sample'] = int(params['reverberation_time'] * params['fs'])
        params['R'] = 1
        params['Z_LOC_SOURCE'] = 1

    return params



def main():
    DATASET_SIZE = 100
    data_type='rir'

    LibriSpeech_PATH = os.path.join(os.getcwd(), 'data')
    DATASET_DEST_PATH = os.path.join(os.getcwd(), 'rir_dataset_generator', 'dev_data')
    Path(DATASET_DEST_PATH).mkdir(parents=True, exist_ok=True)

    dataset_config = get_dataset_params(data_type)

    audio_transformer = torchaudio.transforms.Spectrogram(n_fft=dataset_config['NFFT'], hop_length=dataset_config['HOP_LENGTH'], power=1,
                                                          center=True, pad=0, normalized=True)

    train = torchaudio.datasets.LIBRISPEECH(LibriSpeech_PATH, url='train-clean-100', download=True)
    train_loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=lambda x: data_preprocessing(x, audio_transformer=audio_transformer, **dataset_config))

    theta_array = []

    for i_sample in range(DATASET_SIZE):
        print('Generating sample: ', i_sample)
        (spec_final, sample_rate, theta, winner_est) = next(iter(train_loader))
        filename = os.path.join(DATASET_DEST_PATH, f'{i_sample}.pt')
        theta_array.append(theta)
        scaled = np.int16(spec_final / np.abs(spec_final).max() * 32767)
        winner_est_scaled = np.int16(winner_est / np.abs(winner_est).max() * 32767)
        torch.save((scaled, winner_est_scaled), filename)

    np.save(os.path.join(DATASET_DEST_PATH, 'theta.npy'), np.array(theta_array))
    np.save(os.path.join(DATASET_DEST_PATH, 'dataset_config.npy'), dataset_config)



if __name__ == '__main__':
    main()