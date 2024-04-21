from .modules.residual_stack import ResidualStack
from .modules.jitter import Jitter
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvolutionalDecoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int, use_jitter: bool, jitter_probability: float):

        super(DeconvolutionalDecoder, self).__init__()

        self._use_jitter = use_jitter

        if self._use_jitter:
            self._jitter = Jitter(jitter_probability)

        self._conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self._upsample = nn.Upsample(scale_factor=2)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )

        self._conv_trans_1 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self._conv_trans_2 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self._conv_trans_3 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=out_channels,
            kernel_size=2,
            stride=1,
            padding=0
        )

    def forward(self, inputs):
        # x, BCL
        x = inputs

        if self._use_jitter and self.training:
            x = self._jitter(x)

        x = self._conv_1(x)

        x = self._upsample(x)

        x = self._residual_stack(x)

        x = F.relu(self._conv_trans_1(x))

        x = F.relu(self._conv_trans_2(x))

        x = self._conv_trans_3(x)

        return x
