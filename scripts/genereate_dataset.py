import os.path

import numpy as np
import torchaudio
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from acustic_locating_vq_vae.data_preprocessing import speech_waveform_to_spec, echoed_spec_from_random_rir


def generate_unechoed_spectrogram_dataset(DATASET_SIZE: int, loader: DataLoader, DATASET_DEST_PATH: str,
                                  dataset_config: dict):
    for i_sample in range(DATASET_SIZE):
        waveform, fs, transcript, speaker_id, chapter_id, utterance_id = next(iter(loader))
        spec_final = speech_waveform_to_spec(waveform, dataset_config['fs'], dataset_config['NFFT'],
                                             dataset_config['noverlap'])
        if spec_final is None:
            i_sample -= 1
            continue
        print('Generating sample: ', i_sample)
        filename = os.path.join(DATASET_DEST_PATH, f'{i_sample}.pt')
        i_sample += 1

        torch.save((torch.from_numpy(spec_final), transcript, speaker_id, chapter_id, utterance_id), filename)

    np.save(os.path.join(DATASET_DEST_PATH, 'dataset_config.npy'), dataset_config)


def generate_echoed_spectrogram_dataset(DATASET_SIZE: int, loader: DataLoader, DATASET_DEST_PATH: str,
                                  dataset_config: dict):
    i_sample = 0
    while i_sample < DATASET_SIZE:
        echoed_spec_list, rir_spec_list, sample_rate_list, theta_list, wiener_est_list = next(iter(loader))

        for j_sample in range(len(echoed_spec_list)):
            if i_sample == DATASET_SIZE:
                break
            if echoed_spec_list[j_sample] is None:
                continue
            print('Generating sample: ', i_sample)
            filename = os.path.join(DATASET_DEST_PATH, f'{i_sample}.pt')

            torch.save((echoed_spec_list[j_sample], rir_spec_list[j_sample], sample_rate_list[j_sample],
                        theta_list[j_sample], wiener_est_list[j_sample]), filename)

            i_sample += 1
    np.save(os.path.join(DATASET_DEST_PATH, 'dataset_config.npy'), dataset_config)


def get_dataset_params(data_type: str) -> dict:
    params = {}
    params['fs'] = int(16e3)
    params['NFFT'] = int(2**11)
    params['HOP_LENGTH'] = int(params['fs'] * 0.01)
    olap = 0.75
    params['noverlap'] = round(olap * params['NFFT'])

    if data_type == 'rir' or data_type == 'echoed_speech':
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
    data_type = 'rir'
    dataset_type = 'dev_data'
    rir_batch_size = 20  # For faster dataset generation, in every batch the samples have the same RIR

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

    if data_type == 'rir' or data_type == 'echoed_speech':
        train_loader = DataLoader(librispeech_dataset, batch_size=rir_batch_size, shuffle=True,
                                  collate_fn=lambda x: echoed_spec_from_random_rir(x, **dataset_config))
        generate_echoed_spectrogram_dataset(DATASET_SIZE, train_loader, DATASET_DEST_PATH, dataset_config)
    elif data_type == 'speech':
        train_loader = DataLoader(librispeech_dataset, batch_size=1, shuffle=True)
        generate_unechoed_spectrogram_dataset(DATASET_SIZE, train_loader, DATASET_DEST_PATH, dataset_config)

if __name__ == '__main__':
    main()