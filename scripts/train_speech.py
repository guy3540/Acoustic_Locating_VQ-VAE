import os

import numpy as np
import torch
import torchaudio
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from six.moves import xrange
from line_profiler_pycharm import profile


from acoustic_locating_vq_vae.visualization import plot_spectrogram
from acoustic_locating_vq_vae.vq_vae.convolutional_vq_vae import ConvolutionalVQVAE
from acoustic_locating_vq_vae.data_preprocessing import spec_dataset_preprocessing
from acoustic_locating_vq_vae.rir_dataset_generator.specsdataset import SpecsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASET_PATH = os.path.join(os.getcwd(), "spec_data", "dev_data")
BATCH_SIZE = 64
LR = 1e-3  # as is in the speach article
SAMPLING_RATE = 16e3
NFFT = 400
IN_FEATURE_SIZE = int((NFFT/2) + 1)
# IN_FEATURE_SIZE = 80
HOP_LENGTH = int(SAMPLING_RATE * 0.01)
output_features_dim = IN_FEATURE_SIZE
num_hiddens = 40
in_channels = IN_FEATURE_SIZE
num_residual_layers = 10
num_residual_hiddens = 20
embedding_dim = 40
num_embeddings = 512  # The higher this value, the higher the capacity in the information bottleneck.
commitment_cost = 0.25  # as recommended in VQ VAE article

use_jitter = True
jitter_probability = 0.12

rev = 0.3
olap = 0.75
noverlap = round(olap * NFFT)


@profile
def train(model: ConvolutionalVQVAE, optimizer, num_training_updates):
    model.train()

    train_res_recon_error = []
    train_res_perplexity = []

    # waveform B,C,S
    for i in xrange(num_training_updates):
        (x, _, _, fs, _, _) = next(iter(train_loader))
        x = x.to(device)
        x = torch.abs(x)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        optimizer.zero_grad()
        x = torch.squeeze(x, dim=1)

        vq_loss, reconstructed_x, perplexity = model(x)

        if not x.shape == reconstructed_x.shape:
            reduction = reconstructed_x.shape[2] - x.shape[2]
            reconstructed_x = reconstructed_x[:, :, :-reduction]

        recon_error = F.mse_loss(reconstructed_x, x, reduction='mean')
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 10 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print(f'max in x {torch.max(x).item():.5f}. max recon {torch.max(reconstructed_x).item():.5f} ')
            print(f'min in x {torch.min(x).item():.5f}. min recon {torch.min(reconstructed_x).item():.5f} ')
            print(f'vq loss out of total loss {((vq_loss/loss)*100).item():.5f}')
            print()
        if (i + 1) % 50 == 0:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            plot_spectrogram(torch.hstack((x[0].detach(), reconstructed_x[0].detach())).to('cpu'),
                             title=f"{i} Spectrogram - input", ylabel="freq", ax=ax1)
            freq_to_plot = 10
            ax2.plot(x[0, freq_to_plot, :].detach().to('cpu'), label='input')
            ax2.plot(reconstructed_x[0, freq_to_plot, :].detach().to('cpu'), label="reconstruction")
            ax2.legend()
            ax2.set_title(f'freq{freq_to_plot} ')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('value')

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
    train_dataset = SpecsDataset(root_dir=DATASET_PATH, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: spec_dataset_preprocessing(x))

    model = ConvolutionalVQVAE(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                               commitment_cost, num_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train(model=model, optimizer=optimizer, num_training_updates=15000)
    print("Done")
