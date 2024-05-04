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
            num_hiddens=in_channels,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._pre_vq_conv = nn.Conv1d(
            in_channels=in_channels,
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

