import librosa
import torch
from matplotlib import pyplot as plt

def real_spec_to_complex(spectrogram: torch.Tensor):
    nFeatures = int(spectrogram.shape[0]/2)
    spectrogram_real = spectrogram[0:nFeatures, :]
    spectrogram_complex = spectrogram[nFeatures:, :]
    spec_com = torch.cat((spectrogram_real.unsqueeze(2), spectrogram_complex.unsqueeze(2)), dim=2)
    return torch.view_as_complex(spec_com)

def plot_spectrogram(spectrogram: torch.Tensor, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    if spectrogram.shape[0]== 1 or spectrogram.ndim == 1:
        ax.plot(torch.abs(spectrogram))
    else:
        ax.imshow(librosa.power_to_db(spectrogram.abs()), origin="lower", aspect="auto", interpolation="nearest")

