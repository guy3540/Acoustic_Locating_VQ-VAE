import librosa
from matplotlib import pyplot as plt


def plot_spectrogram(spectrogram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect="auto", interpolation="nearest")
