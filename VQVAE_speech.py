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
from scipy import signal
from scipy.signal.windows import hann

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

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=(3, 2),
                                 stride=1, padding=(1, 0))

        self._residual_stack_1 = Conv1DResidualModel(num_hiddens, num_residual_layers, num_hiddens, embedding_dim)

    def forward(self, inputs):
        x = self._conv_1(inputs).squeeze()
        return self._residual_stack_1(x)


class SpeechDecoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(SpeechDecoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=(3, 2),
                                 stride=1, padding=(1, 1))

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=out_channels,
                                                kernel_size=(4, 1),
                                                stride=(2, 1), padding=(1, 0))

    def forward(self, inputs):
        x = self._conv_1(inputs.unsqueeze(-1))
        x = self._residual_stack(x)
        return self._conv_trans_2(x)


def show(img):
    np_img = img.numpy()
    fig = plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def view_reconstructions(model: VQVAE, dataloader: DataLoader, fs, transform):
    model.eval()

    (originals, _) = next(iter(dataloader))
    if len(originals.shape) > 3:  # Has batches
        originals = originals[0, :].unsqueeze(0)
    originals_complex = torch.complex(real=originals[:, :, :, 0], imag=originals[:, :, :, 1])
    originals = originals.to(device)

    _, reconstructions, _ = model(originals)
    if not reconstructions.shape == originals.shape:
        reconstructions = reconstructions.unsqueeze(0)
    reconstructions_complex = torch.complex(real=reconstructions[:, :, :, 0], imag=reconstructions[:, :, :, 1])
    reconstructions_complex = reconstructions_complex.detach().cpu().numpy()

    fig, axes = plt.subplots(2)
    img_orig = axes[0].pcolormesh(np.arange(originals_complex.shape[2]), transform.f,
                                  20 * np.log10(np.abs(originals_complex.squeeze())), shading='gouraud')

    img_recon = axes[1].pcolormesh(np.arange(reconstructions_complex.shape[2]) * transform.delta_t, transform.f,
                                   20 * np.log10(np.abs(reconstructions_complex.squeeze())), shading='gouraud')

    fig.colorbar(img_orig, ax=axes[0], format='%+2.0f dB')
    fig.colorbar(img_recon, ax=axes[1], format='%+2.0f dB')

    axes[0].set(title='Spectrogram of original signal')
    axes[1].set(title='Spectrogram of reconstructed signal')

    plt.tight_layout()
    plt.show()


def data_preprocessing(data, transform):
    spectrograms = []
    min_size = float("inf")

    for (waveform, sample_rate, _, _, _, _) in data:
        spec_waveform = transform.stft(waveform.numpy())
        real_part = spec_waveform.real
        imag_part = spec_waveform.imag
        spec_waveform = np.stack((real_part, imag_part), axis=-1)
        min_size = min(min_size, spec_waveform.shape[2])
        spectrograms.append(spec_waveform)

    if min_size % 2:
        min_size = min_size - 1

    spectrograms = [torch.from_numpy(spec[:, :, :min_size]).squeeze() for spec in spectrograms]
    spectrograms = torch.stack(spectrograms, 0).to(torch.float)

    return spectrograms, None  # For compatibility with images


def main():
    model_name = "test"
    dataset_path = os.path.join(os.getcwd(), "data")
    batch_size = 1
    val_batch_size = 1
    print_every_n_batches = 100
    n_val_samples_for_eval = val_batch_size
    fs = 16e3
    num_hiddens = 100
    num_residual_hiddens = num_hiddens
    num_residual_layers = 10
    embedding_dim = 128
    num_embeddings = 256
    num_training_updates = 50000
    commitment_cost = 0.25
    learning_rate = 1e-3

    training_data = torchaudio.datasets.LIBRISPEECH(dataset_path, url='train-clean-100', download=True)

    train_set, val_set = torch.utils.data.random_split(training_data, [0.9, 0.1])

    w = hann(int(fs * 0.025))
    transform = signal.ShortTimeFFT(w, hop=int(fs * 0.01), fs=fs)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: data_preprocessing(x, transform))
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True,
                            collate_fn=lambda x: data_preprocessing(x, transform))

    num_f = next(iter(train_loader))[0].shape[1]

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

    torch.save(model.state_dict(), 'models/' + model_name + '_st.pt')
    torch.save(model, 'models/' + model_name + '.pt')

    model.plot_losses()
    plt.show()

    view_reconstructions(model, val_loader, fs, transform)


if __name__ == "__main__":
    main()
