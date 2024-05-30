import os.path

import numpy as np
import scipy.signal as ss
import rir_generator as rir
import torchaudio
import torch
from torch.utils.data import DataLoader
from pathlib import Path


def convert_speech_to_specs(data, fixed_rir, fixed_speech):
    if fixed_rir:
        theta = convert_speech_to_specs.theta
    else:
        theta = torch.from_numpy(np.random.uniform(low=-np.pi, high=np.pi, size=1))

    z_loc = np.array([Z_LOC_SOURCE])
    h_src_loc = np.stack((R * np.cos(theta), R * np.sin(theta), z_loc), axis=1) + receiver_position
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
        if fixed_speech and len(convert_speech_to_specs.speech) == 0:
            convert_speech_to_specs.speech = waveform
        elif fixed_speech:
            waveform = convert_speech_to_specs.speech

        speech_spec = np.squeeze(audio_transformer(waveform))
        waveform_h = ss.convolve(waveform.squeeze(), h_RIR.squeeze(), mode='same')
        echoed_spec = audio_transformer(torch.from_numpy(waveform_h))

        rir_spec = np.divide(speech_spec, echoed_spec + 1e-8)
        rir_spec = np.divide(rir_spec, np.abs(rir_spec).max())

        wiener_est = (torch.sum(echoed_spec * np.conjugate(speech_spec), dim=1) /
                      (torch.sum(speech_spec * np.conjugate(speech_spec), dim=1) + 1e-8))
        wiener_est = wiener_est.abs().pow(2)
        rir_spec = rir_spec.abs().pow(2)
        speech_spec = speech_spec.abs().pow(2)
        echoed_spec = echoed_spec.abs().pow(2)

    return speech_spec, rir_spec, echoed_spec, sample_rate, theta, wiener_est


if __name__ == '__main__':
    C = 340
    fs = 16e3
    receiver_position = [2.5, 1.5, 1.5]
    room_dimensions = [4, 5, 3]
    reverberation_time = 0.4
    n_sample = int(reverberation_time * fs)
    R = 1
    DATASET_SIZE = 1000
    Z_LOC_SOURCE = 1

    fixed_rir = False
    fixed_speech = False

    convert_speech_to_specs.theta = torch.from_numpy(np.random.uniform(low=-np.pi, high=np.pi, size=1))
    convert_speech_to_specs.speech = []

    LibriSpeech_PATH = os.path.join(os.getcwd(), 'data')
    DATASET_DEST_PATH = os.path.join(os.getcwd(), 'spec_data', 'test')
    Path(DATASET_DEST_PATH).mkdir(parents=True, exist_ok=True)

    NFFT = int(fs * 0.025)
    HOP_LENGTH = int(fs * 0.01)

    dataset_config = {
        "fs": int(fs),
        "receiver_position": receiver_position,
        "room_dimensions": room_dimensions,
        "reverberation_time": reverberation_time,
        "n_sample": n_sample,
        "R": R,
        "NFFT": NFFT,
        "HOP_LENGTH": HOP_LENGTH,
        "Z_LOC_SOURCE": Z_LOC_SOURCE
    }

    audio_transformer = torchaudio.transforms.Spectrogram(n_fft=NFFT, hop_length=HOP_LENGTH, power=None,
                                                          center=True, pad=0, normalized=True)

    train = torchaudio.datasets.LIBRISPEECH(LibriSpeech_PATH, url='train-clean-100', download=True)
    train_loader = DataLoader(train, batch_size=1, shuffle=True,
                              collate_fn=lambda x: convert_speech_to_specs(x, fixed_rir, fixed_speech))

    for i_sample in range(DATASET_SIZE):
        print('Generating sample: ', i_sample)
        (speech_spec, rir_spec, echoed_spec, sample_rate, theta, wiener_est) = next(iter(train_loader))
        filename = os.path.join(DATASET_DEST_PATH, f'{i_sample}.pt')
        torch.save((speech_spec, rir_spec, echoed_spec, sample_rate, theta, wiener_est), filename)

    np.save(os.path.join(DATASET_DEST_PATH, 'dataset_config.npy'), dataset_config)
