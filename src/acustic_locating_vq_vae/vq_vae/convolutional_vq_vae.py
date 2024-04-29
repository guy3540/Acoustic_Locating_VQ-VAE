import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from six.moves import xrange

from acustic_locating_vq_vae.vq_vae.convolutional_encoder import ConvolutionalEncoder
from acustic_locating_vq_vae.vq_vae.deconvolutional_decoder import DeconvolutionalDecoder
from acustic_locating_vq_vae.vq_vae.vector_quantizer import VectorQuantizer

import torch.nn as nn
import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvolutionalVQVAE(nn.Module):

    def __init__(self, in_channels: int, num_hiddens: int, embedding_dim: int, num_residual_layers: int, num_residual_hiddens: int,
                 commitment_cost: float, num_embeddings: int, use_jitter: bool=True, encoder_average_pooling: bool=False, out_channels: int = None):
        out_channels = in_channels if out_channels is None else out_channels

        super(ConvolutionalVQVAE, self).__init__()
        self.encoder_average_pooling = encoder_average_pooling
        self._encoder = ConvolutionalEncoder(
            in_channels=in_channels,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._pre_vq_conv = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1
        )

        self._vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self._decoder = DeconvolutionalDecoder(
            in_channels=embedding_dim,
            out_channels=out_channels,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_jitter=use_jitter,
            jitter_probability=0.25,
        )

    def get_embedding_dim(self):
        return self._vq.get_embedding_dim()

    def train_on_data(self, optimizer: optim, dataloader: DataLoader, num_training_updates, data_variance):
        self.train()
        train_res_recon_error = []
        train_res_perplexity = []

        inputs: torch.Tensor
        labels: torch.Tensor

        for i in xrange(num_training_updates):
            (inputs, _) = next(iter(dataloader))
            inputs = inputs.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = self(inputs)
            if not inputs.shape == data_recon.shape:
                recon_error = F.mse_loss(data_recon, inputs[:, :, :-1]) / data_variance
            else:
                recon_error = F.mse_loss(data_recon, inputs) / data_variance
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

        self.train_res_recon_error = train_res_recon_error
        self.train_res_perplexity = train_res_perplexity

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        # if self.encoder_average_pooling:
        #     z = torch.mean(z, dim=2, keepdim=True)
        loss, quantized, perplexity, _ = self._vq(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

    def get_latent_representation(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        return self._vq(z)

