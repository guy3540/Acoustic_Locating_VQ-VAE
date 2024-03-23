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

from convolutional_vq_vae import ConvolutionalVQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = r"C:\Users\reiem\PycharmProjects\Acoustic_Locating_VQ-VAE\BasedOnVQ-VAE-Speech\modules\data"
BATHC_SIZE = 1
LR = 4e-4  # as is in the speach article
NFFT = 512
IN_FEACHER_SIZE = int((NFFT/2) + 1)

configuration = {}

# CONV VQVAE
configuration['augment_output_features'] = True
configuration['output_features_dim'] = IN_FEACHER_SIZE
configuration['decay'] = 0.0
#
# #CONV ENC
configuration['num_hiddens'] = 768
configuration['input_features_dim'] = IN_FEACHER_SIZE
configuration['num_residual_layers'] = 2
configuration['use_kaiming_normal'] = False  #todo remove this for we dont know what this is
configuration['augment_input_features'] = True
configuration['sampling_rate'] = 16000
#
# #PRE_VQ_CON
# configuration['num_hiddens']
configuration['embedding_dim'] = 64
#
# #VQ
configuration['num_embeddings'] = 512  # The higher this value, the higher the capacity in the information bottleneck.
# configuration['embedding_dim']
configuration['commitment_cost'] = 0.25  # as recommended in VQ VAE article
#
#
# #CONV DECODER
# configuration['embedding_dim']
# out_channels = self._output_features_filters,
# configuration['num_hiddens']
# configuration['num_residual_layers']
configuration['residual_channels'] = 768
# configuration['use_kaiming_normal']
configuration['use_jitter'] = True
configuration['jitter_probability'] = 0.12
configuration['use_speaker_conditioning'] = False

audio_transformer = torchaudio.transforms.Spectrogram(n_fft=NFFT)


def data_preprocessing(data):
    spectrograms = []
    for (waveform, _, _, _, _, _) in data:
        spec = audio_transformer(waveform).squeeze(0).transpose(0,1)
        spectrograms.append(spec)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

    return spectrograms#, sample_rate, transcript, speaker_id, chapter_id, utterance_id


train = torchaudio.datasets.LIBRISPEECH(DATASET_PATH, url='train-clean-100', download=True)
train_loader = DataLoader(train, batch_size=BATHC_SIZE, shuffle=False, collate_fn=lambda x: data_preprocessing(x))




def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def train(model: ConvolutionalVQVAE, optimizer):
    model.train()
    spectrogram = T.Spectrogram(n_fft=NFFT).to(device)
    train_res_recon_error = []
    train_res_perplexity = []
    # waveform B,C,S
    for batch, x in enumerate(train_loader):
        #waveform size N,C,L -> N,L (C=1) dosent go well with 1d conv
        # waveform = waveform.to(device)
        # # x size B,C,n_fft // 2 + 1, len(waveform)/hop_length
        # x = spectrogram(waveform)
        x = x.to(device)
        x = torch.squeeze(x, dim=1)
        reconstructed_x, vq_loss, perplexity = model(x)

        recon_error = F.mse_loss(reconstructed_x, x)  #/ data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (batch + 1) % 100 == 0:
            print('%d iterations' % (batch + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()

            fig, (ax1, ax2) = plt.subplots(1,2)
            plot_spectrogram(x[0].detach().to('cpu'), title="Spectrogram - input", ylabel="freq", ax=ax1)
            plot_spectrogram(reconstructed_x[0].detach().to('cpu'), title="Spectrogram - reconstructed", ylabel="freq", ax=ax2)
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


model = ConvolutionalVQVAE(configuration=configuration, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train(model=model, optimizer=optimizer)
print("init")
