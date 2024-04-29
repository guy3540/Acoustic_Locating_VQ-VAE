import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from six.moves import xrange
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from acustic_locating_vq_vae.rir_dataset_generator.rir_dataset import RIR_DATASET
from acustic_locating_vq_vae.vq_vae.deconvolutional_decoder import DeconvolutionalDecoder
from acustic_locating_vq_vae.visualization import plot_spectrogram
from train_rir import rir_data_preprocessing


rir_model = torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_rir.pt'))
speech_model = torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_speech.pt'))

BATCH_SIZE = 64
num_training_updates = 100
num_hiddens = 50
num_residual_layers = 2
num_residual_hiddens = 40
use_jitter = True
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class acoustic_location_model(nn.Module):
    def __init__(self, rir_model, speech_model, out_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, use_jitter):
        super(acoustic_location_model, self).__init__()

        self.rir_model = rir_model.to(device)
        self.speech_model = speech_model.to(device)

        self.embedding_dim = self.rir_model._vq._embedding_dim + self.speech_model._vq._embedding_dim

        self._decoder = DeconvolutionalDecoder(
            in_channels=self.embedding_dim,
            out_channels=out_channels,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_jitter=use_jitter,
            jitter_probability=0.25,
        )

    def forward(self, spec_in):
        _, rir_quantized, rir_perplexity, _ = self.rir_model.get_latent_representation(spec_in)

        rir_quantized = rir_quantized[:, :, 0].squeeze()  # TODO Delete once we have an updated model

        _, speech_quantized, speech_perplexity, _ = self.speech_model.get_latent_representation(spec_in)

        rir_quantized_rep = rir_quantized.unsqueeze(2).repeat(1, 1, speech_quantized.shape[2])

        stacked_res = torch.zeros((rir_quantized_rep.shape[0],
                                   rir_quantized_rep.shape[1]*2, rir_quantized_rep.shape[2]), device=device)
        stacked_res[:, ::2, :] = rir_quantized_rep.detach()
        stacked_res[:, 1::2, :] = speech_quantized.detach()

        return self._decoder(stacked_res), speech_perplexity, rir_perplexity


DATASET_PATH = os.path.join(os.path.dirname(__file__), 'rir_dataset_generator', 'dev_data')
train_data = RIR_DATASET(DATASET_PATH)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda datum: rir_data_preprocessing(datum))

sample_to_init, source_coordinates, mic, room, fs = next(iter(train_loader))
out_channels = sample_to_init.shape[1]

model = acoustic_location_model(rir_model, speech_model, out_channels, num_hiddens, num_residual_layers,
                                num_residual_hiddens, use_jitter).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)

train_res_recon_error = []
train_speech_perp = []
train_rir_perp = []

model.train()
for i in xrange(num_training_updates):
    x, source_coordinates, mic, room, fs = next(iter(train_loader))
    x = x.type(torch.FloatTensor)
    x = x.to(device)
    x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)

    optimizer.zero_grad()
    reconstructed_x, speech_perplexity, rir_perplexity = model(x)

    if not x.shape == reconstructed_x.shape:
        reduction = reconstructed_x.shape[2] - x.shape[2]
        recon_error = F.mse_loss(reconstructed_x[:, :, :-reduction], x)
    else:
        recon_error = F.mse_loss(reconstructed_x, x)

    loss = recon_error
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(loss.item())
    train_speech_perp.append(speech_perplexity.item())
    train_rir_perp.append(rir_perplexity.item())

    if (i + 1) % 100 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('speech perplexity: %.3f' % np.mean(train_speech_perp[-100:]))
        print('rir perplexity: %.3f' % np.mean(train_rir_perp[-100:]))
    if (i + 1) % 500 == 0:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot_spectrogram(torch.squeeze(x[0]).detach().to('cpu'), title="Spectrogram - input", ylabel="mag", ax=ax1)
        plot_spectrogram(torch.squeeze(reconstructed_x[0]).detach().to('cpu'), title="Spectrogram - reconstructed",
                         ylabel="mag",
                         ax=ax2)
        plt.show()
