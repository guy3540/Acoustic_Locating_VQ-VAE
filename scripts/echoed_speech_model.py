import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from six.moves import xrange
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from acustic_locating_vq_vae.rir_dataset_generator.rir_dataset import RIR_DATASET
from acustic_locating_vq_vae.visualization import plot_spectrogram
from train_rir import rir_data_preprocessing, rir_data_preprocess_permute_normalize_and_cut


rir_model = torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_rir.pt'))
speech_model = torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_speech.pt'))

BATCH_SIZE = 64
num_training_updates = 10000
num_hiddens = 80
num_residual_layers = 2
num_residual_hiddens = 80
use_jitter = True
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EchoedSpeechReconModel(nn.Module):
    def __init__(self, rir_model, speech_model, out_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, use_jitter):
        super(EchoedSpeechReconModel, self).__init__()

        self.rir_model = rir_model.to(device)
        self.speech_model = speech_model.to(device)

        self.rir_model._vq.set_train_vq(False)
        self.speech_model._vq.set_train_vq(False)

        self.embedding_dim = self.rir_model.get_embedding_dim()  #+ self.speech_model.get_embedding_dim()

    def forward(self, spec_in, spec_in_rir):
        _, rir_quantized, rir_perplexity, _ = self.rir_model.get_latent_representation(spec_in_rir)

        _, speech_quantized, speech_perplexity, _ = self.speech_model.get_latent_representation(spec_in)

        ## Assume that speech_quantized is [Batch_Size, embedding_dim, t]
        ## Assume that rir_quantized is [Batch_Size, embedding_dim, 1]
        rir_quantized = torch.mean(rir_quantized, dim=2).unsqueeze(2)
        #######

        quantized = speech_quantized * rir_quantized  # quantized shape is the same as speech_quantized

        return self.speech_model._decoder(quantized), speech_perplexity, rir_perplexity


DATASET_PATH = os.path.join(os.path.dirname(__file__), 'train_data')
train_data = RIR_DATASET(DATASET_PATH)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda datum: rir_data_preprocessing(datum))

sample_to_init, _, _, _, _, _ = next(iter(train_loader))
out_channels = sample_to_init.shape[1]

model = EchoedSpeechReconModel(rir_model, speech_model, out_channels, num_hiddens, num_residual_layers,
                               num_residual_hiddens, use_jitter).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)

train_res_recon_error = []
train_speech_perp = []
train_rir_perp = []

model.train()
for i in xrange(num_training_updates):
    x, wiener_est, source_coordinates, mic, room, fs = next(iter(train_loader))
    x = x.type(torch.FloatTensor)
    x = x.to(device)
    x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)

    x_rir, wiener_est, source_coordinates, mic, room, fs = rir_data_preprocess_permute_normalize_and_cut(
        (x, wiener_est, source_coordinates, mic, room, fs))

    optimizer.zero_grad()
    reconstructed_x, speech_perplexity, rir_perplexity = model(x, x_rir.to(device))

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
        print('==========================================')
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

f = plt.figure(figsize=(16, 8))
ax = f.add_subplot(1, 1, 1)
ax.plot(train_res_recon_error)
ax.set_yscale('log')
ax.set_title('Smoothed NMSE.')
ax.set_xlabel('iteration')

plt.show()


torch.save(model, '../models/model_echoed_speech.pt')
