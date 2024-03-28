 #####################################################################################
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
 #   OUT OF OR IN CONNECTION+ WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from modules.residual_stack import ResidualStack
from modules.jitter import Jitter
# from speech_utils.global_conditioning import GlobalConditioning
# from error_handling.console_logger import ConsoleLogger

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvolutionalDecoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers,
        num_residual_hiddens, use_jitter, jitter_probability,
        use_speaker_conditioning):

        super(DeconvolutionalDecoder, self).__init__()

        self._use_jitter = use_jitter
        self._use_speaker_conditioning = use_speaker_conditioning

        if self._use_jitter:
            self._jitter = Jitter(jitter_probability)

        # FIXME hardcoded
        in_channels = in_channels + 40 if self._use_speaker_conditioning else in_channels

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

    def forward(self, inputs, speaker_dic=None, speaker_id=None):
        #x, BCL
        x = inputs

        if self._use_jitter and self.training:
            x = self._jitter(x)

        if self._use_speaker_conditioning:
            speaker_embedding = GlobalConditioning.compute(speaker_dic, speaker_id, x,
                device=self._device, gin_channels=40, expand=True)
            x = torch.cat([x, speaker_embedding], dim=1).to(self._device)

        x = self._conv_1(x)


        x = self._upsample(x)

        
        x = self._residual_stack(x)

        
        x = F.relu(self._conv_trans_1(x))


        x = F.relu(self._conv_trans_2(x))


        x = self._conv_trans_3(x)

        
        return x
