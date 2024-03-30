import glob
import os
import re

import numpy as np
from scipy.io.wavfile import read

import torch
from torch.utils.data import Dataset


class RIR_DATASET(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_path = glob.glob(os.path.join(self.root_dir, 'rir_*.wav'))

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, idx):
        wav_path = self.dataset_path[idx]
        sample_rate, wav = read(wav_path)
        wav_data = torch.from_numpy(np.array(wav, dtype=np.float32))
        room = self.get_room_dimensions(wav_path)
        mic = self.get_mic_location(wav_path)
        source_location = self.get_source_location(wav_path)

        return wav_data, source_location, mic, room, sample_rate

    def get_room_dimensions(self, path):
        """
        Extracts the three integers after the letter "R" from a path string.
        """
        match = re.search(r"R_(\d+)_(\d+)_(\d+)", path)
        if match:
            return [int(group) for group in match.groups()]
        else:
            return None

    def get_mic_dimensions(self, path):
        """
        Extracts the three integers after the letter "R" from a path string.
        """
        match = re.search(r"M_(\d+)_(\d+)_(\d+)", path)
        if match:
            return [int(group) for group in match.groups()]
        else:
            return None

    def get_mic_location(self, path):
        """
        Extracts the three integers after the letter "R" from a path string.
        """
        match = re.search(r"M_(\d+)_(\d+)_(\d+)", path)
        if match:
            return [int(group) for group in match.groups()]
        else:
            return None

    def get_source_location(self, path):
        """
      Extracts the three floats after the letter "S" from a path string.
      """
        match = re.search(r"S_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)", path)
        if match:
            return [float(group) for group in match.groups()]
        else:
            return None
