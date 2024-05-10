import os
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from six.moves import xrange

from acoustic_locating_vq_vae.data_preprocessing import rir_data_preprocessing
from acoustic_locating_vq_vae.rir_dataset_generator.rir_dataset import RIR_DATASET
from acoustic_locating_vq_vae.visualization import plot_spectrogram

from acoustic_locating_vq_vae.vq_vae.convolutional_vq_vae import ConvolutionalVQVAE
from scripts.train_location import run_location_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vq_vae(model: ConvolutionalVQVAE, optimizer, train_loader, num_training_updates):
    model.train()

    train_res_recon_error = []
    train_res_perplexity = []

    # waveform B,C,S
    for i in xrange(num_training_updates):
        x, winner_est, source_coordinates, mic, room, fs = next(iter(train_loader))
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = torch.permute(x, [0, 2, 1])
        winner_est = winner_est.type(torch.FloatTensor)
        winner_est = (winner_est - torch.mean(winner_est, dim=1, keepdim=True)) / (torch.std(winner_est, dim=1, keepdim=True) + 1e-8)
        winner_est = torch.unsqueeze(winner_est,1)
        winner_est = winner_est.to(device)

        optimizer.zero_grad()
        vq_loss, reconstructed_x, perplexity = model(x)

        if not winner_est.shape == reconstructed_x.shape:
            reduction = reconstructed_x.shape[2] - winner_est.shape[2]
            recon_error = F.mse_loss(reconstructed_x[:, :, :-reduction], winner_est)  # / data_variance
        else:
            recon_error = F.mse_loss(reconstructed_x, winner_est)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
        if (i + 1) % 500 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plot_spectrogram(torch.squeeze(x[0]).detach().to('cpu'), title="Spectrogram - input", ylabel="mag", ax=ax1)
            ax2.plot(torch.squeeze(winner_est[0]).detach().to('cpu'), label="Winner Estimate")
            ax2.plot(torch.squeeze(reconstructed_x[0]).detach().to('cpu'), label="reconstruction")
            ax2.legend()
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
    torch.save(model, 'model_rir.pt')
    torch.save(model.state_dict(), 'model_rir_state_dict.pt')


def run_rir_training():
    DATASET_PATH = Path(os.getcwd()) / 'rir_dataset_generator' / 'dev_data'
    BATCH_SIZE = 64
    LR = 1e-3
    IN_FEATURE_SIZE = 500
    num_training_updates = 15000

    # CONV VQVAE
    num_hiddens = 40
    in_channels = IN_FEATURE_SIZE
    out_channels = 1
    num_residual_layers = 2
    num_residual_hiddens = 20
    embedding_dim = 40
    num_embeddings = 512  # The higher this value, the higher the capacity in the information bottleneck.
    commitment_cost = 0.25  # as recommended in VQ VAE article
    use_jitter = False

    train_data = RIR_DATASET(DATASET_PATH)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: rir_data_preprocessing(x))

    model = ConvolutionalVQVAE(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                               commitment_cost, num_embeddings, use_jitter=use_jitter, encoder_average_pooling=True,
                               out_channels=out_channels
                               ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train_vq_vae(model=model, optimizer=optimizer, train_loader=train_loader, num_training_updates=num_training_updates)


if __name__ == '__main__':
    run_rir_training()
