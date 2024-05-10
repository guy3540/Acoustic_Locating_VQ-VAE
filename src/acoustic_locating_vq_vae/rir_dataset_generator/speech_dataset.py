import glob
import os
import numpy as np

import torch
from torch.utils.data import Dataset


class speech_DATASET(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.dataset_files = glob.glob(os.path.join(self.root_dir, '*.pt'))

        dataset_config = np.load(os.path.join(root_dir, 'dataset_config.npy'),
                                 allow_pickle=True).item()

        self.fs = dataset_config['fs']
        self.NFFT = dataset_config['NFFT']
        self.HOP_LENGTH = dataset_config['HOP_LENGTH']

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        item_filename = "{}.pt".format(idx)
        item_path = os.path.join(self.root_dir, item_filename)
        spectrogram, transcript, speaker_id, chapter_id, utterance_id = torch.load(item_path)

        return spectrogram, transcript, speaker_id, chapter_id, utterance_id, self.fs