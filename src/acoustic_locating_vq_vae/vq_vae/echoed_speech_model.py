import torch
from torch import nn
from torch.nn import functional as F

from acoustic_locating_vq_vae.vq_vae.deconvolutional_decoder import DeconvolutionalDecoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EchoedSpeechReconModel(nn.Module):
    def __init__(self, rir_model, speech_model, out_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, use_jitter):
        super(EchoedSpeechReconModel, self).__init__()

        self.rir_model = rir_model.to(device)
        self.speech_model = speech_model.to(device)

        self.rir_model._vq.set_train_vq(False)
        self.speech_model._vq.set_train_vq(False)
        self.flag_train_encoder = False

        self.embedding_dim = self.rir_model.get_embedding_dim() + self.speech_model.get_embedding_dim()

        self._decoder = DeconvolutionalDecoder(
            in_channels=self.embedding_dim,
            out_channels=out_channels,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_jitter=use_jitter,
            jitter_probability=0.25,
        )

    def set_train_encoder(self, flag):
        self.flag_train_encoder = flag

    def forward(self, spec_in, spec_in_rir):
        _, rir_quantized, rir_perplexity, _ = self.rir_model.get_latent_representation(spec_in_rir)

        _, speech_quantized, speech_perplexity, _ = self.speech_model.get_latent_representation(spec_in)

        size_diff = speech_quantized.size(2) - rir_quantized.size(2)

        # Pad rir_quantized tensor
        if size_diff > 0:
            # Calculate pad width
            pad_width = (0, size_diff)  # Pad only along the third dimension

            # Pad tensor
            rir_quantized = F.pad(rir_quantized, pad_width)

        if self.flag_train_encoder:
            quantized = torch.cat((speech_quantized, rir_quantized), dim=1)
        else:
            quantized = torch.cat((speech_quantized.detach(), rir_quantized.detach()), dim=1)

        return self._decoder(quantized), speech_perplexity, rir_perplexity
