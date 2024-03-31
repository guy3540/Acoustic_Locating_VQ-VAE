from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy.signal import savgol_filter
from six.moves import xrange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, flag_flatten=True):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.flag_flatten = flag_flatten

    def forward(self, inputs):
        input_shape = inputs.shape

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
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.contiguous(), perplexity, encodings


class VQVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 num_embeddings: int, embedding_dim: int, commitment_cost: float, flag_flatten: bool = True):
        super(VQVAE, self).__init__()

        self._encoder = encoder
        self._vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, flag_flatten)
        self._decoder = decoder

        self.train_res_perplexity = []
        self.train_res_recon_error = []
        self.train_error_on_val_example = []

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
        x_recon = self.decode(quantized)
        return loss, x_recon, perplexity

    def train_on_data(self, optimizer: optim, dataloader: DataLoader, num_training_updates, data_variance,
                      val_loader, print_every_n_batches=1, n_val_samples_for_eval=10):
        self.train()
        train_res_recon_error = []
        train_res_perplexity = []
        train_error_on_val_example = []

        inputs: torch.Tensor
        labels: torch.Tensor

        for i in xrange(num_training_updates):
            (inputs, _) = next(iter(dataloader))
            inputs = inputs.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = self(inputs)
            if not data_recon.shape == inputs.shape:
                data_recon = data_recon.unsqueeze(0)
            recon_error = F.mse_loss(data_recon, inputs, reduction='mean') / data_variance
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if (i+1) % print_every_n_batches == 0:
                with torch.no_grad():
                    self.eval()
                    loss = 0
                    for ind in range(n_val_samples_for_eval):
                        (val_inputs, _) = next(iter(val_loader))
                        val_inputs = val_inputs.to(device)
                        vq_loss, data_recon, perplexity = self(val_inputs)
                        if not data_recon.shape == val_inputs.shape:
                            data_recon = data_recon.unsqueeze(0)
                        recon_error = F.mse_loss(data_recon, val_inputs) / data_variance
                        loss += recon_error + vq_loss
                    loss = loss/n_val_samples_for_eval
                    train_error_on_val_example.append(loss.item())


                print('%d batches' % (i + 1))
                print('train recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('train perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
                print('validation examples recon_error: %.3f' % loss.item())
                print()
                self.train()

        self.train_res_recon_error = train_res_recon_error
        self.train_res_perplexity = train_res_perplexity
        self.train_error_on_val_example = train_error_on_val_example
        self.print_every_n_batches = print_every_n_batches

    def plot_losses(self):
        if not self.train_res_recon_error:  # Return if is empty
            return [], []
        # train_res_recon_error_smooth = savgol_filter(self.train_res_recon_error, len(self.train_res_recon_error), 7)
        # train_res_perplexity_smooth = savgol_filter(self.train_res_perplexity, len(self.train_res_perplexity), 7)
        # train_error_on_val_example_smooth = savgol_filter(self.train_error_on_val_example,
        #                                                   len(self.train_error_on_val_example), 7)
        fig, axs = plt.subplots(3, 1, sharex=True)
        ax = axs[0]
        ax.plot(self.train_res_recon_error)
        ax.set_yscale('log')
        ax.set_title('Smoothed Normalized MSE.')
        ax.set_xlabel('iteration')

        ax = axs[1]
        ax.plot(self.train_res_perplexity)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')

        ax = axs[2]
        iter_grid = [self.print_every_n_batches*(i + 1) for i in range(len(self.train_error_on_val_example))]
        ax.plot(iter_grid, self.train_error_on_val_example)
        ax.set_title('Smoothed loss on an example from validation set')
        ax.set_xlabel('iteration')
        return fig, ax
