import torch
from torch.nn import functional as F
import os
from pathlib import Path

from torch.utils.data import DataLoader

from acoustic_locating_vq_vae.data_preprocessing import spec_dataset_preprocessing
from acoustic_locating_vq_vae.rir_dataset_generator.specsdataset import SpecsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model_on_data(model, data_loader, dataset_size):
    model.eval()

    res = []

    for i in range(dataset_size):
        (x, winner_est, source_coordinates, mic, room, fs) = next(iter(data_loader))
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = torch.unsqueeze(x, 1)

        loss, quantized, perplexity, encodings = model.get_latent_representation(x)
        encodings = encodings.view(x.size(0), quantized.size(2), encodings.size(1))
        encodings = encodings.view(x.size(0), quantized.size(2) * encodings.size(2))
        location = model(encodings)

        loss = F.mse_loss(location, source_coordinates.float())
        loss_list.append(loss.item())



if __name__ == '__main__':
    original_rir_model = torch.load('../models/model_echoed_speech_6500.pt').to(device)
    encoder_trained_model = torch.load('../models/model_echoed_trained_encoders_3000.pt')

    BATCH_SIZE = 64
    VAL_DATASET_PATH = Path(os.getcwd()) / 'spec_data' / 'val_set'
    val_data = SpecsDataset(VAL_DATASET_PATH)

    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=lambda x: spec_dataset_preprocessing(x))

    dataset_size = len(val_data)
