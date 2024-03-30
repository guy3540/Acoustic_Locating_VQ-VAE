import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from six.moves import xrange
from VQVAE_speech import \
    SpeechEncoder, SpeechDecoder  #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from convolutional_encoder import ConvolutionalEncoder
from deconvolutional_decoder import DeconvolutionalDecoder
from vector_quantizer import VectorQuantizer
from vector_quantizer_ema import VectorQuantizerEMA
# from error_handling.console_logger import ConsoleLogger


import torch.nn as nn
import torch
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConvolutionalVQVAE(nn.Module):

    def __init__(self,in_channels, num_hiddens,embedding_dim,num_residual_layers,num_residual_hiddens,commitment_cost, num_embeddings, use_jitter=True):
        super(ConvolutionalVQVAE, self).__init__()
        # self._encoder = SpeechEncoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)

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
        # self._decoder = SpeechDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, in_channels)
        self._decoder = DeconvolutionalDecoder(
            in_channels=embedding_dim,
            out_channels=in_channels,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_jitter=use_jitter,
            jitter_probability=0.25,
            use_speaker_conditioning=False,
        )
    def train_on_data(self, optimizer: optim, dataloader: DataLoader, num_training_updates, data_variance):
        self.train()
        train_res_recon_error = []
        train_res_perplexity = []

        inputs: torch.Tensor
        labels: torch.Tensor

        for i in xrange(num_training_updates):
            (inputs,_) = next(iter(dataloader))
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

            if (i+1) % 100 == 0:
                print('%d iterations' % (i + 1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
                print()

        self.train_res_recon_error = train_res_recon_error
        self.train_res_perplexity = train_res_perplexity
    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity
