import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from six.moves import xrange

from acustic_locating_vq_vae.data_preprocessing import combine_tensors_with_min_dim
from acustic_locating_vq_vae.rir_dataset_generator.rir_dataset import RIR_DATASET
from acustic_locating_vq_vae.visualization import plot_spectrogram

from acustic_locating_vq_vae.vq_vae.convolutional_vq_vae import ConvolutionalVQVAE
from acustic_locating_vq_vae.vq_vae.location_model.location_model import LocationModule
from acustic_locating_vq_vae.data_preprocessing import rir_data_preprocess_permute_normalize_and_cut
from acustic_locating_vq_vae.data_preprocessing import rir_data_preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class rir_model(ConvolutionalVQVAE):
    def __init__(self, in_channels: int, num_hiddens: int, embedding_dim: int, num_residual_layers: int,
                 num_residual_hiddens: int, commitment_cost: float, num_embeddings: int, use_jitter: bool = True,
                 encoder_average_pooling: bool = False, out_channels: int = None):
        super().__init__(in_channels, num_hiddens, embedding_dim, num_residual_layers,num_residual_hiddens,
                         commitment_cost, num_embeddings, use_jitter, encoder_average_pooling, out_channels)

        self.time_pool = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = torch.mean(x, dim=2).unsqueeze(1)  # Collapse time axis
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq(z)
        x_recon = self._decoder(quantized)
        x_recon = self.time_pool(x_recon)
        return loss, x_recon, perplexity



def train_vq_vae(model: ConvolutionalVQVAE, optimizer, train_loader, num_training_updates):
    model.train()

    train_res_recon_error = []
    train_res_perplexity = []

    # waveform B,C,S
    for i in xrange(num_training_updates):
        x, wiener_est, source_coordinates, mic, room, fs = next(iter(train_loader))
        # x, wiener_est, source_coordinates, mic, room, fs = (
        #     rir_data_preprocess_permute_normalize_and_cut((x, wiener_est, source_coordinates, mic, room, fs)))

        x = x.to(device)
        wiener_est = wiener_est.to(torch.float).to(device)

        optimizer.zero_grad()
        vq_loss, reconstructed_x, perplexity = model(x)
        reconstructed_x = reconstructed_x.squeeze(2)

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
            plot_spectrogram(torch.squeeze(wiener_est[0]).detach().to('cpu'), title="Spectrogram - input", ylabel="mag", ax=ax1)
            plot_spectrogram(torch.squeeze(reconstructed_x[0]).detach().to('cpu'), title="Spectrogram - reconstructed",
                             ylabel="mag",
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
    torch.save(model, 'model_rir.pt')
    torch.save(model.state_dict(), 'model_rir_state_dict.pt')


def run_rir_training():
    DATASET_PATH = Path(os.getcwd()) / 'rir_dataset_generator' / 'dev_data'
    BATCH_SIZE = 64
    LR = 1e-3
    IN_FEATURE_SIZE = 201
    num_training_updates = 10000

    # CONV VQVAE
    num_hiddens = 40
    in_channels = 201
    out_channels = IN_FEATURE_SIZE
    num_residual_layers = 2
    num_residual_hiddens = 20
    embedding_dim = 40
    num_embeddings = 512  # The higher this value, the higher the capacity in the information bottleneck.
    commitment_cost = 0.25  # as recommended in VQ VAE article
    use_jitter = False

    train_data = RIR_DATASET(DATASET_PATH)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: rir_data_preprocessing(x))

    model = rir_model(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                      commitment_cost, num_embeddings, use_jitter=use_jitter, encoder_average_pooling=True,
                      out_channels=out_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train_vq_vae(model=model, optimizer=optimizer, train_loader=train_loader, num_training_updates=num_training_updates)


if __name__ == '__main__':
    run_rir_training()
    # run_location_training()
