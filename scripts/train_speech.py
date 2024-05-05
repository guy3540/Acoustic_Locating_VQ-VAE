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
from acustic_locating_vq_vae.visualization import plot_spectrogram, real_spec_to_complex
from acustic_locating_vq_vae.vq_vae.convolutional_vq_vae import ConvolutionalVQVAE
from acustic_locating_vq_vae.data_preprocessing import batchify_spectrograms
from acustic_locating_vq_vae.rir_dataset_generator.speech_dataset import speech_DATASET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = os.path.join(os.getcwd(), "speech_dataset", "train_data")
BATCH_SIZE = 16
LR = 1e-4  # as is in the speach article
SAMPLING_RATE = 16e3
NFFT = int(2**11)
IN_FEATURE_SIZE = int((NFFT) + 2)
# IN_FEATURE_SIZE = 80
HOP_LENGTH = int(SAMPLING_RATE * 0.01)
output_features_dim = IN_FEATURE_SIZE
num_hiddens = 1024
in_channels = IN_FEATURE_SIZE
num_residual_layers = 10
num_residual_hiddens = 1024
embedding_dim = 512
num_embeddings = 128  # The higher this value, the higher the capacity in the information bottleneck.
commitment_cost = 0.25  # as recommended in VQ VAE article

use_jitter = True
jitter_probability = 0.12

rev = 0.3
olap = 0.75
noverlap = round(olap * NFFT)


def train(model: ConvolutionalVQVAE, optimizer, num_training_updates):
    model.train()

    train_res_recon_error = []
    train_res_perplexity = []
    train_vq_loss = []

    # waveform B,C,S
    for i in xrange(num_training_updates):
        (x, sample_rate) = next(iter(train_loader))
        x = x.to(device)

        optimizer.zero_grad()
        x = torch.squeeze(x, dim=1)
        vq_loss, reconstructed_x, perplexity = model(x)

        if not x.shape == reconstructed_x.shape:
            retuction = reconstructed_x.shape[2] - x.shape[2]
            print("reduction is: " + retuction)
            recon_error = F.mse_loss(reconstructed_x[:, :, :-retuction], x, reduction='mean')
        else:
            recon_error = F.mse_loss(reconstructed_x, x, reduction='mean')
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        train_vq_loss.append(vq_loss.item())

        if (i + 1) % 10 == 0:
            print('%d iterations' % (i + 1))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print('recon_error: %.6f' % np.mean(train_res_recon_error[-100:]))
            print('train_vq_loss: %.6f' % np.mean(train_vq_loss[-100:]))
            print('total loss: %.3f' % (np.mean(train_vq_loss[-100:]) +
                  np.mean(train_res_recon_error[-100:])))
            print()
        if (i + 1) % 500 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plot_spectrogram(real_spec_to_complex(x[0].detach().to('cpu')), title=f"{i} Spectrogram - input", ylabel="freq", ax=ax1)
            plot_spectrogram(real_spec_to_complex(reconstructed_x[0].detach().to('cpu')), title="Spectrogram - reconstructed", ylabel="freq",
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


if __name__ == '__main__':
    train_dataset = speech_DATASET(root_dir=DATASET_PATH, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: batchify_spectrograms(x, NFFT, noverlap))

    model = ConvolutionalVQVAE(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                               commitment_cost, num_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train(model=model, optimizer=optimizer, num_training_updates=150000)
    # model.train_on_data(optimizer,train_loader,num_training_updates=15000, data_variance=1)
    print("init")