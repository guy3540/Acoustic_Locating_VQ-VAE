import os

import numpy as np
import torch
import torchaudio
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from six.moves import xrange

from acustic_locating_vq_vae.data_preprocessing import combine_tensors_with_min_dim
from acustic_locating_vq_vae.visualization import plot_spectrogram
from acustic_locating_vq_vae.vq_vae.convolutional_vq_vae import ConvolutionalVQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASET_PATH = os.path.join(os.getcwd(), "data")
BATCH_SIZE = 64
LR = 1e-3  # as is in the speach article
SAMPLING_RATE = 16e3
NFFT = int(SAMPLING_RATE * 0.025)
IN_FEATURE_SIZE = int((NFFT / 2) + 1)
# IN_FEATURE_SIZE = 80
HOP_LENGTH = int(SAMPLING_RATE * 0.01)
output_features_dim = IN_FEATURE_SIZE
num_hiddens = 40
in_channels = IN_FEATURE_SIZE
num_residual_layers = 10
num_residual_hiddens = 20
embedding_dim = 40
num_embeddings = 1024  # The higher this value, the higher the capacity in the information bottleneck.
commitment_cost = 0.25  # as recommended in VQ VAE article

use_jitter = True
jitter_probability = 0.12


audio_transformer = torchaudio.transforms.Spectrogram(n_fft=NFFT, hop_length=HOP_LENGTH,
                                                      power=1, center=True, pad=0, normalized=True)
# audio_transformer = torchaudio.transforms.MelSpectrogram(n_fft=NFFT, sample_rate=SAMPLING_RATE,hop_length=HOP_LENGTH,n_mels=IN_FEATURE_SIZE)
# audio_transformer = torchaudio.transforms.MelSpectrogram(n_fft=NFFT, sample_rate=SAMPLING_RATE,hop_length=HOP_LENGTH,n_mels=IN_FEATURE_SIZE, window_fn=torch.hann_window, power=1.0, center=True)


def speech_data_preprocessing(data):
    spectrograms = []
    for (waveform, sample_rate, _, _, _, _) in data:
        spec = audio_transformer(waveform)
        spectrograms.append(spec)

    spectrograms = combine_tensors_with_min_dim(spectrograms)

    return spectrograms, sample_rate,  # transcript, speaker_id, chapter_id, utterance_id


def train(model: ConvolutionalVQVAE, optimizer, num_training_updates):
    model.train()

    train_res_recon_error = []
    train_res_perplexity = []

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # waveform B,C,S
    for i in xrange(num_training_updates):
        (x, _) = next(iter(train_loader))
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = x.to(device)

        optimizer.zero_grad()
        x = torch.squeeze(x, dim=1)
        vq_loss, reconstructed_x, perplexity = model(x)

        if not x.shape == reconstructed_x.shape:
            retuction = reconstructed_x.shape[2] - x.shape[2]
            recon_error = F.mse_loss(reconstructed_x[:, :, :-retuction], x)  # / data_variance
        else:
            recon_error = F.mse_loss(reconstructed_x, x)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 10 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
        if (i + 1) % 1000 == 0:
            plot_spectrogram(x[0].detach().to('cpu'), title="Spectrogram - input", ylabel="freq", ax=ax1)
            plot_spectrogram(reconstructed_x[0].detach().to('cpu'), title="Spectrogram - reconstructed", ylabel="freq",
                             ax=ax2)
            plt.show()

    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')
    plt.show()
    torch.save(model, '../models/model_speech.pt')
    torch.save(model.state_dict(), '../models/model_speech_state_dict.pt')

if __name__ == '__main__':
    train_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_PATH, url='train-clean-100', download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: speech_data_preprocessing(x))

    model = ConvolutionalVQVAE(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                               commitment_cost, num_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train(model=model, optimizer=optimizer, num_training_updates=15000)
    # model.train_on_data(optimizer,train_loader,num_training_updates=15000, data_variance=1)
    print("init")
