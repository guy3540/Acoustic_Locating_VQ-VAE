import torch
import os

from acustic_locating_vq_vae.rir_dataset_generator.rir_dataset import RIR_DATASET

dataset = RIR_DATASET(root_dir=os.path.join(os.getcwd(), 'rir_dataset_generator', 'dev_data'))

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

for i, (wav_data, source_location, mic, room, sample_rate) in enumerate(loader):

    print(i)