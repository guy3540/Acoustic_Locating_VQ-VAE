from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy.signal import savgol_filter
from six.moves import xrange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = f.mse_loss(quantized.detach(), inputs)
        q_latent_loss = f.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.contiguous(), perplexity, encodings


class VQVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VQVAE, self).__init__()

        self._encoder = encoder
        self._vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = decoder

        self.train_res_perplexity = []
        self.train_res_recon_error = []

    def encode(self, x):
        return self._encoder(x)

    def decode(self, quantized):
        return self._decoder(quantized)

    def quantize_latent(self, z):
        loss, quantized, perplexity, encodings = self._vq(z)
        return loss, quantized, perplexity, encodings

    def forward(self, x):
        z = self.encode(x)
        loss, quantized, perplexity, _ = self.quantize_latent(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

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
            recon_error = f.mse_loss(data_recon, inputs) / data_variance
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

    def plot_losses(self):
        if not self.train_res_recon_error:  # Return if is empty
            return [], []
        train_res_recon_error_smooth = savgol_filter(self.train_res_recon_error, len(self.train_res_recon_error), 7)
        train_res_perplexity_smooth = savgol_filter(self.train_res_perplexity, len(self.train_res_perplexity), 7)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed Normalized MSE.')
        ax.set_xlabel('iteration')

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(train_res_perplexity_smooth)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')
        return f, ax
