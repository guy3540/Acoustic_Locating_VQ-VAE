from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from VQVAE_class import VQVAE
from Res_classes import Residual, ResidualStack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder_Images(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder_Images, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._pre_vq = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=embedding_dim,
                                 kernel_size=1,
                                 stride=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)
        return self._pre_vq(x)


class Decoder_Images(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder_Images, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


def show(img):
    np_img = img.numpy()
    fig = plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def view_reconstructions(model: VQVAE, dataloader: DataLoader):
    model.eval()

    (originals, _) = next(iter(dataloader))
    originals = originals.to(device)

    _, reconstructions, _ = model(originals)

    show(make_grid(reconstructions.cpu().data) + 0.5, )
    plt.show()
    show(make_grid(originals.cpu()+0.5))
    plt.show()


def main():
    batch_size = 256
    num_training_updates = 15000

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 64
    num_embeddings = 512

    commitment_cost = 0.25

    learning_rate = 1e-3

    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                       ]))

    data_variance = np.var(training_data.data / 255.0)

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True, pin_memory=True)

    encoder = Encoder_Images(in_channels=3, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
                      num_residual_hiddens=num_residual_hiddens, embedding_dim=embedding_dim)

    decoder = Decoder_Images(in_channels=embedding_dim, num_hiddens=num_hiddens,
                      num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)

    model = VQVAE(encoder=encoder, decoder=decoder, num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                  commitment_cost=commitment_cost).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.train_on_data(optimizer, training_loader, num_training_updates, data_variance)

    model.plot_losses()

    plt.show()

    view_reconstructions(model, validation_loader)


if __name__ == "__main__":
    main()
