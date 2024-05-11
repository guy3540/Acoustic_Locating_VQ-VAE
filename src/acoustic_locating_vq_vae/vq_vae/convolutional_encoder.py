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
        nn.init.kaiming_uniform_(self._conv_1.weight, a=0, mode="fan_in", nonlinearity="relu")


        self._conv_2 = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )
        nn.init.kaiming_uniform_(self._conv_2.weight, a=0, mode="fan_in", nonlinearity="relu")

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 =  nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=2
        )
        nn.init.kaiming_uniform_(self._conv_3.weight, a=0, mode="fan_in", nonlinearity="relu")

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )
        nn.init.kaiming_uniform_(self._conv_4.weight, a=0, mode="fan_in", nonlinearity="relu")

        self._conv_5 = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )
        nn.init.kaiming_uniform_(self._conv_5.weight, a=0, mode="fan_in", nonlinearity="relu")

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

        x_conv_1 = F.relu(self._conv_1(inputs))

        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        
        x_conv_3 = F.relu(self._conv_3(x))

        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3

        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4

        x = self._residual_stack(x_conv_5) + x_conv_5

        return x