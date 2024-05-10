import os
from pathlib import Path

import numpy as np
import torch
from line_profiler import profile
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from six.moves import xrange

from acoustic_locating_vq_vae.data_preprocessing import spec_dataset_preprocessing
from acoustic_locating_vq_vae.rir_dataset_generator.specsdataset import SpecsDataset
from acoustic_locating_vq_vae.visualization import plot_spectrogram

from acoustic_locating_vq_vae.vq_vae.convolutional_vq_vae import ConvolutionalVQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@profile
def train_vq_vae(model: ConvolutionalVQVAE, optimizer, train_loader, num_training_updates):
    model.train()

    train_res_recon_error = []
    train_res_perplexity = []

    # waveform B,C,S
    for i in xrange(num_training_updates):
        _, rir_spec, _, _, _, wiener_est = next(iter(train_loader))
        x = rir_spec.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = torch.permute(x, [0, 2, 1])
        wiener_est = wiener_est.type(torch.FloatTensor)
        wiener_est = (wiener_est - torch.mean(wiener_est, dim=1, keepdim=True)) / (torch.std(wiener_est, dim=1, keepdim=True) + 1e-8)
        wiener_est = torch.unsqueeze(wiener_est, 1)
        wiener_est = wiener_est.to(device)

        optimizer.zero_grad()
        vq_loss, reconstructed_x, perplexity = model(x)

        if not wiener_est.shape == reconstructed_x.shape:
            reduction = reconstructed_x.shape[2] - wiener_est.shape[2]
            recon_error = F.mse_loss(reconstructed_x[:, :, :-reduction], wiener_est)  # / data_variance
        else:
            recon_error = F.mse_loss(reconstructed_x, wiener_est)
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
            ax2.plot(torch.squeeze(wiener_est[0]).detach().to('cpu'), label="Winner Estimate")
            ax2.plot(torch.squeeze(reconstructed_x[0]).detach().to('cpu'), label="reconstruction")
            ax2.legend()
            plt.show()

    train_res_recon_error_smooth = train_res_recon_error
    train_res_perplexity_smooth = train_res_perplexity

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
    torch.save(model, '../models/model_rir.pt')


def run_rir_training():
    DATASET_PATH = Path(os.getcwd()) / 'spec_data' / 'dev_data'
    BATCH_SIZE = 64
    LR = 1e-3
    IN_FEATURE_SIZE = 500
    num_training_updates = 150

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

    train_data = SpecsDataset(DATASET_PATH)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: spec_dataset_preprocessing(x))

    model = ConvolutionalVQVAE(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                               commitment_cost, num_embeddings, use_jitter=use_jitter, out_channels=out_channels
                               ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train_vq_vae(model=model, optimizer=optimizer, train_loader=train_loader, num_training_updates=num_training_updates)


if __name__ == '__main__':
    run_rir_training()
