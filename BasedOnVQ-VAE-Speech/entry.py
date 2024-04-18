import os

import numpy as np
import torch
import torchaudio
from scipy.signal import savgol_filter
from torch import nn
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
from six.moves import xrange

import rir_generator as rir
import scipy.signal as ss

from convolutional_vq_vae import ConvolutionalVQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = '/home/guy/PycharmProjects/Acoustic_Locating_VQ-VAE/data'
BATHC_SIZE = 64
LR = 1e-3  # as is in the speach article
SAMPLING_RATE = 16e3
NFFT = int(SAMPLING_RATE * 0.025)
IN_FEACHER_SIZE = int((NFFT / 2) + 1)
# IN_FEACHER_SIZE = 80
HOP_LENGTH = int(SAMPLING_RATE * 0.01)

# CONV VQVAE
output_features_dim = IN_FEACHER_SIZE
#
# #CONV ENC
num_hiddens = 40
in_channels = IN_FEACHER_SIZE
num_residual_layers = 10
num_residual_hiddens = 20

#
# #PRE_VQ_CON
embedding_dim = 40
#
# #VQ
num_embeddings = 1024  # The higher this value, the higher the capacity in the information bottleneck.
commitment_cost = 0.25  # as recommended in VQ VAE article
#
#
# #CONV DECODER
use_jitter = True
jitter_probability = 0.12
use_speaker_conditioning = False

audio_transformer = torchaudio.transforms.Spectrogram(n_fft=NFFT, hop_length=HOP_LENGTH, power=1, center=True, pad=0, normalized=True)
# audio_transformer = torchaudio.transforms.MelSpectrogram(n_fft=NFFT, sample_rate=SAMPLING_RATE,hop_length=HOP_LENGTH,n_mels=IN_FEACHER_SIZE)
# audio_transformer = torchaudio.transforms.MelSpectrogram(n_fft=NFFT, sample_rate=SAMPLING_RATE,hop_length=HOP_LENGTH,n_mels=IN_FEACHER_SIZE, window_fn=torch.hann_window, power=1.0, center=True)

h_c = 340
h_rec_pos = [2, 1, 1]  # When calculating the RIR, for now, we assume 1m distance
h_room_dim = [4, 5, 3]
h_rev_time = 0.4
h_n_sample = int(h_rev_time * SAMPLING_RATE)

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
    thetas = []

    theta = np.random.uniform(low=-np.pi, high=np.pi, size=1)
    thetas.append(theta)
    z_loc = np.random.uniform(low=0, high=1, size=1)
    h_src_loc = np.stack((np.cos(theta).T, np.sin(theta).T, z_loc.T), axis=1) + h_rec_pos
    h = rir.generate(
        c=h_c,  # Sound velocity (m/s)
        fs=SAMPLING_RATE,  # Sample frequency (samples/s)
        r=h_rec_pos,
        s=np.squeeze(h_src_loc),  # Source position [x y z] (m)
        L=h_room_dim,  # Room dimensions [x y z] (m)
        reverberation_time=h_rev_time,  # Reverberation time (s)
        nsample=h_n_sample,  # Number of output samples
    )

    for (waveform, sample_rate, _, _, _, _) in data:

        spec_signal = audio_transformer(waveform)

        waveform_h = ss.convolve(waveform.squeeze(), h.squeeze(), mode='same')

        spec_with_h = audio_transformer(torch.from_numpy(waveform_h))

        spec = np.divide(spec_signal, spec_with_h)
        spec = np.divide(spec, np.abs(spec).max())

        spectrograms.append(spec)

    spectrograms = combine_tensors_with_min_dim(spectrograms)
    # spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    # labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, sample_rate,  # transcript, speaker_id, chapter_id, utterance_id

#(B,F,T)
# def data_preprocessing(data):
#     spectrograms = []
#     for (waveform, sample_rate, _, _, _, _) in data:
#         # Convert waveform to spectrogram
#         # Extract log Mel-filterbanks
#         mel_spec = librosa.feature.melspectrogram(
#             y=waveform[0].numpy(),
#             sr=SAMPLING_RATE,
#             n_fft=int(SAMPLING_RATE * 0.025),  # window size of 25 ms
#             hop_length=int(SAMPLING_RATE * 0.01),  # step size of 10 ms
#             n_mels=80,
#             norm=None,
#             power=1.0
#         )
#         if mel_spec.shape[1] % 2 != 0:
#             mel_spec = mel_spec[:,:-1]
#         spectrograms.append(torch.from_numpy(mel_spec).unsqueeze(dim=0))
#     spectrograms = combine_tensors_with_min_dim(spectrograms)
#     return spectrograms, None  # For compatibility with images


train = torchaudio.datasets.LIBRISPEECH(DATASET_PATH, url='train-clean-100', download=True)
train_loader = DataLoader(train, batch_size=BATHC_SIZE, shuffle=True, collate_fn=lambda x: data_preprocessing(x))


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
        (x, _) = next(iter(train_loader))
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

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
        if (i + 1) % 100 == 0:
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
    torch.save(model, 'model.pt')

if __name__ == '__main__':

    model = ConvolutionalVQVAE(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                               commitment_cost, num_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train(model=model, optimizer=optimizer, num_training_updates=15000)
    # model.train_on_data(optimizer,train_loader,num_training_updates=15000, data_variance=1)
    print("init")
