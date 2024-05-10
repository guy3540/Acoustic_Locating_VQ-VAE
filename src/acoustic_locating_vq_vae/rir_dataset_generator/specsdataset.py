import glob
import os
import numpy as np

import torch
from torch.utils.data import Dataset


class SpecsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_files = glob.glob(os.path.join(self.root_dir, '*.pt'))

        dataset_config = np.load(os.path.join(root_dir, 'dataset_config.npy'),
                                 allow_pickle=True).item()

        self.fs = dataset_config['fs']
        self.receiver_position = dataset_config['receiver_position']
        self.room_dimensions = dataset_config['room_dimensions']
        self.reverberation_time = dataset_config['reverberation_time']
        self.n_sample = dataset_config['n_sample']
        self.R = dataset_config['R']
        self.NFFT = dataset_config['NFFT']
        self.HOP_LENGTH = dataset_config['HOP_LENGTH']
        self.Z_LOC_SOURCE = dataset_config['Z_LOC_SOURCE']

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        item_filename = "{}.pt".format(idx)
        item_path = os.path.join(self.root_dir, item_filename)
        speech_spec, rir_spec, echoed_spec, sample_rate, theta, wiener_est = torch.load(item_path)

        return speech_spec, rir_spec, echoed_spec, sample_rate, theta, wiener_est

    def get_source_coordinates(self, theta):

        z_loc = np.array([self.Z_LOC_SOURCE])
        receiver_position = self.receiver_position
        h_src_loc = (receiver_position +
                     np.stack((self.R * np.cos(theta).T, self.R * np.sin(theta).T, z_loc), axis=1))
        h_src_loc = np.minimum(h_src_loc, self.room_dimensions)
        return h_src_loc
