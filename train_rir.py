import os

import numpy as np
import torch
import torchaudio
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
from six.moves import xrange
from rir_dataset_generator.rir_dataset import RIR_DATASET
import Utilities

from convolutional_vq_vae import ConvolutionalVQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

git_root_path = Utilities.get_git_root()
DATASET_PATH = os.path.join(git_root_path, 'rir_dataset_generator', 'rir_dataset')
BATCH_SIZE = 1
LR = 1e-3
SAMPLING_RATE = 16e3
NFFT = int(SAMPLING_RATE * 0.025)
IN_FEATURE_SIZE = int((NFFT / 2) + 1)
HOP_LENGTH = int(SAMPLING_RATE * 0.01)

# CONV VQVAE
output_features_dim = IN_FEATURE_SIZE
num_hiddens = 40
in_channels = IN_FEATURE_SIZE
num_residual_layers = 10
num_residual_hiddens = 20
embedding_dim = 3
num_embeddings = 5  # The higher this value, the higher the capacity in the information bottleneck.
commitment_cost = 0.25  # as recommended in VQ VAE article
use_jitter = False
jitter_probability = 0.12

audio_transformer = torchaudio.transforms.Spectrogram(n_fft=NFFT, hop_length=HOP_LENGTH, power=1, center=True, pad=0, normalized=True)


def combine_tensors_with_min_dim(tensor_list):
    """
  Combines a list of PyTorch tensors with shapes (1, H, x1), (1, H, x2), ..., (1, H, xN)
  into a new tensor of shape (N, H, X), where X is the minimum dimension among x1, x2, ..., xN.

  Args:
      tensor_list: A list of PyTorch tensors with the same height (H).

  Returns:
      A new tensor of shape (N, H, X), where X is the minimum dimension.

  Raises:
      ValueError: If the tensors in the list do not have the same height (H).
  """

    if not tensor_list:
        raise ValueError("Input tensor list cannot be empty")

    # Check if all tensors have the same height (H)
    H = tensor_list[0].size(1)
    for tensor in tensor_list:
        if tensor.size(1) != H:
            raise ValueError("All tensors in the list must have the same height (H)")

    # Get the minimum dimension (X) across all tensors in the list
    min_dim = min(tensor.size(2) for tensor in tensor_list)

    # Create a new tensor to store the combined data
    combined_tensor = torch.zeros((len(tensor_list), H, min_dim))

    # Fill the combined tensor with data from the input tensors, selecting the minimum value for each element
    for i, tensor in enumerate(tensor_list):
        combined_tensor[i, :, :] = tensor[:, :, :min_dim]

    return combined_tensor


def data_preprocessing(data):
    spectrograms = []
    source_locations =[]
    mic_locations = []
    room_dimentrions = []
    sample_rates =[]
    for (waveform, source_location, mic, room, sample_rate) in data:
        spec = audio_transformer(waveform)
        spectrograms.append(torch.unsqueeze(spec,dim=0))
        source_locations.append(source_location)
        mic_locations.append(mic)
        room_dimentrions.append(room)
        sample_rates.append(sample_rate)

    spectrograms = combine_tensors_with_min_dim(spectrograms)

    return spectrograms, torch.tensor(source_locations)


train = RIR_DATASET(DATASET_PATH)
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: data_preprocessing(x))


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def train(model: ConvolutionalVQVAE, optimizer, num_training_updates):
    model.train()

    train_res_recon_error = []
    train_res_perplexity = []

    # waveform B,C,S
    for i in xrange(num_training_updates):
        (x, source_location) = next(iter(train_loader))
        x = x.to(device)

        optimizer.zero_grad()
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

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
        if (i + 1) % 500 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
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
    torch.save(model, 'model_rir.pt')

if __name__ == '__main__':

    model = ConvolutionalVQVAE(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                               commitment_cost, num_embeddings, use_jitter=use_jitter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train(model=model, optimizer=optimizer, num_training_updates=15000)
    # model.train_on_data(optimizer,train_loader,num_training_updates=15000, data_variance=1)
    print("init")
