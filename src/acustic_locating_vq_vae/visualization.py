import librosa
import torch
from matplotlib import pyplot as plt


def plot_spectrogram(spectrogram: torch.Tensor, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    if spectrogram.shape[0]== 1 or spectrogram.ndim == 1:
        ax.plot(torch.abs(spectrogram))
    else:
        ax.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect="auto", interpolation="nearest")

