import torch
import Utilities
import os

from rir_dataset import RIR_DATASET

git_root_path = Utilities.get_git_root()
dataset = RIR_DATASET(root_dir=os.path.join(git_root_path, 'rir_dataset_generator', 'rir_dataset'))
git_root_path = Utilities.get_git_root()
LibriSpeech_PATH = os.path.join(git_root_path, 'data')
DATASET_DEST_PATH = os.path.join(git_root_path, 'rir_dataset_generator', 'dev_data')

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

for i, (wav_data, source_location, mic, room, sample_rate) in enumerate(loader):

    print("i")