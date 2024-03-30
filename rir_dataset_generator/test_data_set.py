import torch

from rir_dataset import RIR_DATASET


dataset = RIR_DATASET(root_dir=r"C:\Users\reiem\PycharmProjects\Acoustic_Locating_VQ-VAE\rir_dataset_generator\rir_dataset")

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

for i, (wav_data, source_location, mic, room, sample_rate) in enumerate(loader):

    print("i")