from .modules.residual_stack import ResidualStack

import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalEncoder(nn.Module):
    
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):

        super(ConvolutionalEncoder, self).__init__()

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self._relu = nn.ReLU()

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )


    def forward(self, inputs):

        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        return x
