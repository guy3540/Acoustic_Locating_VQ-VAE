import os.path

import numpy as np
import scipy.signal as ss
import rir_generator as rir
import torchaudio
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from acustic_locating_vq_vae.data_preprocessing import speech_waveform_to_spec

def rir_data_preprocessing(data, Z_LOC_SOURCE, R, room_dimensions, receiver_position, fs, reverberation_time,
                           n_sample, audio_transformer, C, **kwargs):
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

        wiener_est = torch.sum(spec_with_h * np.conjugate(spec_signal), dim=1) / (
                    torch.sum(spec_signal * np.conjugate(spec_signal), dim=1) + 1e-8)

    return spec_final, sample_rate, theta, wiener_est  # transcript, speaker_id, chapter_id, utterance_id


def get_dataset_params(data_type: str) -> dict:
    params = {}
    params['fs'] = int(16e3)
    params['NFFT'] = int(2**11)
    params['HOP_LENGTH'] = int(params['fs'] * 0.01)
    olap = 0.75
    params['noverlap'] = round(olap * params['NFFT'])

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
    DATASET_SIZE = 10000
    data_type = 'speech'
    dataset_type = 'train_data'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LibriSpeech_PATH = os.path.join(os.getcwd(), 'data')
    if data_type == 'rir':
        DATASET_DEST_PATH = os.path.join(os.getcwd(), 'rir_dataset_generator', dataset_type)
    elif data_type == 'speech':
        DATASET_DEST_PATH = os.path.join(os.getcwd(), 'speech_dataset', dataset_type)
    elif data_type == 'echoed_speech':
        DATASET_DEST_PATH = os.path.join(os.getcwd(), 'echoed_speech_dataset', dataset_type)
    else:
        raise ValueError('Illegal data type')

    Path(DATASET_DEST_PATH).mkdir(parents=True, exist_ok=True)

    dataset_config = get_dataset_params(data_type)

    librispeech_dataset = torchaudio.datasets.LIBRISPEECH(LibriSpeech_PATH, url='train-clean-100', download=True)

    if data_type == 'rir':
        train_loader = DataLoader(librispeech_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: rir_data_preprocessing(x, audio_transformer=audio_transformer, **dataset_config))
        theta_array = []
    elif data_type == 'speech':
        train_loader = DataLoader(librispeech_dataset, batch_size=1, shuffle=True)
    elif data_type == 'echoed_speech':
        train_loader = DataLoader(librispeech_dataset, batch_size=1, shuffle=True)

    for i_sample in range(DATASET_SIZE):
        print('Generating sample: ', i_sample)
        if data_type == 'rir':
            (spec_final, sample_rate, theta, winner_est) = next(iter(train_loader))
        elif data_type == 'speech':
            waveform, fs, transcript, speaker_id, chapter_id, utterance_id = next(iter(train_loader))
            spec_final = speech_waveform_to_spec(waveform, dataset_config['fs'], dataset_config['NFFT'],
                                                 dataset_config['noverlap'])
        filename = os.path.join(DATASET_DEST_PATH, f'{i_sample}.pt')

        if data_type == 'rir':
            theta_array.append(theta)
            wiener_est_scaled = np.int16(winner_est / np.abs(winner_est).max() * 32767)
            torch.save((spec_final, wiener_est_scaled), filename)
        elif data_type == 'speech':
            torch.save((torch.from_numpy(spec_final)), filename)

    if data_type == 'rir':
        np.save(os.path.join(DATASET_DEST_PATH, 'theta.npy'), np.array(theta_array))
    np.save(os.path.join(DATASET_DEST_PATH, 'dataset_config.npy'), dataset_config)



if __name__ == '__main__':
    main()