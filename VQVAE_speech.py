from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from VQVAE_class import VQVAE
from Res_classes import Residual, ResidualStack, Conv1DResidualModel

import os
import urllib.request
import tarfile
import torchaudio
import librosa

from scipy import signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpeechEncoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(SpeechEncoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack_1 = Conv1DResidualModel(num_hiddens, num_residual_layers, num_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        return self._residual_stack_1(x)


class SpeechDecoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(SpeechDecoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens,
                                             dims=1)

        self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        return self._conv_trans_2(x)


def show(img):
    np_img = img.numpy()
    fig = plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def view_reconstructions(model: VQVAE, dataloader: DataLoader, fs):
    model.eval()

    (originals, _) = next(iter(dataloader))
    originals_db = librosa.power_to_db(originals, ref=np.max)
    originals = originals.to(device)

    _, reconstructions, _ = model(originals)
    reconstructions = librosa.power_to_db(reconstructions.detach().cpu(), ref=np.max)

    fig, axes = plt.subplots(2)

    img = librosa.display.specshow(originals_db.squeeze(), x_axis='time',
                                   y_axis='mel', sr=fs,
                                   fmax=8000, ax=axes[0])

    fig.colorbar(img, ax=axes[0], format='%+2.0f dB')
    axes[0].set(title='Mel-frequency spectrogram')

    img2 = librosa.display.specshow(reconstructions.squeeze(), x_axis='time',
                                    y_axis='mel', sr=fs,
                                    fmax=8000, ax=axes[1])

    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    axes[1].set(title='Mel-frequency spectrogram')

    plt.show()


def data_preprocessing(data, sample_rate):
    spectrograms = []
    min_size = float("inf")
    for (waveform, sample_rate, _, _, _, _) in data:
        # Convert waveform to spectrogram
        # Extract log Mel-filterbanks
        mel_spec = librosa.feature.melspectrogram(
            y=waveform.numpy(),
            sr=sample_rate,
            n_fft=int(sample_rate * 0.025),  # window size of 25 ms
            hop_length=int(sample_rate * 0.01),  # step size of 10 ms
            n_mels=80,
            norm=None,
            power=1.0
        )
        min_size = min(min_size, mel_spec.shape[2])
        spectrograms.append(mel_spec)

    if min_size % 2:
        min_size = min_size - 1

    spectrograms = [torch.from_numpy(spec[:, :, :min_size]).squeeze() for spec in spectrograms]
    spectrograms = torch.stack(spectrograms, 0)

    return spectrograms, None  # For compatibility with images


def main():
    dataset_path = os.path.join(os.getcwd(), "data")
    batch_size = 20
    val_batch_size = 1
    print_every_n_batches = 1
    n_val_samples_for_eval = 10
    fs = 16e3
    num_hiddens = 40
    num_residual_hiddens = 20
    num_residual_layers = 10
    embedding_dim = 40
    num_embeddings = 1024
    num_training_updates = 10
    commitment_cost = 0.25
    learning_rate = 1e-3
    num_f = 80

    training_data = torchaudio.datasets.LIBRISPEECH(dataset_path, url='train-clean-100', download=True)

    train_set, val_set = torch.utils.data.random_split(training_data, [0.9, 0.1])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: data_preprocessing(x, fs))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True,
                            collate_fn=lambda x: data_preprocessing(x, fs))

    data_variance = 1

    encoder = SpeechEncoder(in_channels=num_f, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
                            num_residual_hiddens=num_residual_hiddens, embedding_dim=embedding_dim)

    decoder = SpeechDecoder(in_channels=embedding_dim, num_hiddens=num_hiddens,
                            num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens,
                            out_channels=num_f)

    model = VQVAE(encoder=encoder, decoder=decoder, num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                  commitment_cost=commitment_cost, flag_flatten=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.train_on_data(optimizer, train_loader, num_training_updates, data_variance, val_loader,
                        print_every_n_batches, n_val_samples_for_eval)

    model.plot_losses()
    plt.show()

    view_reconstructions(model, val_loader, fs)

    torch.save(model.state_dict(), 'models/model_300k_iters_st.pt')
    torch.save(model, 'models/model_300k_iters.pt')


if __name__ == "__main__":
    main()
