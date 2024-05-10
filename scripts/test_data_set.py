import torch
import os

from acoustic_locating_vq_vae.rir_dataset_generator.specsdataset import SpecsDataset

dataset = SpecsDataset(root_dir=os.path.join(os.getcwd(), 'rir_dataset_generator', 'dev_data'))

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

for i, (wav_data, source_location, mic, room, sample_rate) in enumerate(loader):

    print(i)