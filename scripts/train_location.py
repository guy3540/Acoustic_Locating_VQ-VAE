import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from six.moves import xrange
from torch.nn import functional as F
from torch.utils.data import DataLoader

from acoustic_locating_vq_vae.data_preprocessing import spec_dataset_preprocessing
from acoustic_locating_vq_vae.rir_dataset_generator.specsdataset import SpecsDataset
from acoustic_locating_vq_vae.vq_vae.location_model.location_model import LocationModule
from acoustic_locating_vq_vae.vq_vae.echoed_speech_model import EchoedSpeechReconModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_samples_test_on_validation_set = 500


def run_location_training():
    DATASET_PATH = Path(os.getcwd()) / 'spec_data' / '10k_set'
    VAL_DATASET_PATH = Path(os.getcwd()) / 'spec_data' / 'val_set'
    encoder_output_dim = 101
    embedding_dim = 40
    BATCH_SIZE = 64
    num_training_updates = 15000
    train_percent = 0.95

    train_data = SpecsDataset(DATASET_PATH)
    val_data = SpecsDataset(VAL_DATASET_PATH)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: spec_dataset_preprocessing(x))

    val_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: spec_dataset_preprocessing(x))

    vae_model = torch.load("/home/guy/PycharmProjects/Acoustic_Locating_VQ-VAE/models/model_echoed_speech.pt").to(device)
    location_model = LocationModule(encoder_output_dim, embedding_dim, 1).to(device)
    optimizer = torch.optim.Adam(location_model.parameters(), lr=1e-3)

    train_location(vae_model, location_model, optimizer, num_training_updates, train_loader, val_loader)


def evaluate_location_model(test_data, location_model):
    BATCH_SIZE = 2
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    vae_model = torch.load('../models/model_rir.pt').to(device)

    vae_model.eval()
    location_model.eval()
    loss_list = []
    for i, (x, winner_est, source_coordinates, mic, room, fs) in enumerate(test_loader):
        source_coordinates = torch.squeeze(source_coordinates).to(device)
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = torch.unsqueeze(x, 1)

        loss, quantized, perplexity, encodings = vae_model.get_latent_representation(x)
        encodings = encodings.view(x.size(0), quantized.size(2), encodings.size(1))
        encodings = encodings.view(x.size(0), quantized.size(2) * encodings.size(2))
        location = location_model(encodings)

        loss = F.mse_loss(location, source_coordinates.float())
        loss_list.append(loss.item())

    mean_loss = torch.mean(torch.as_tensor(loss_list))
    return mean_loss


def train_location(combined_model: EchoedSpeechReconModel, location_model, optimizer, num_training_updates, train_loader,
                   val_loader):
    combined_model.eval()
    location_model.train()

    train_location_error = []
    test_location_error = []
    val_error = []
    last_error_val_test = float('inf')

    # waveform B,C,S
    for i in xrange(num_training_updates):

        if (i + 1) % n_samples_test_on_validation_set == 0:  # Test on validation test for early stopping
            location_model.eval()
            _, _, echoed_specs, _, theta, _ = next(iter(val_loader))
        else:
            _, _, echoed_specs, _, theta, _ = next(iter(train_loader))
        x = echoed_specs.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = x.permute(0, 2, 1)

        optimizer.zero_grad()
        loss, quantized, perplexity, encodings = combined_model.rir_model.get_latent_representation(x)
        # encodings = encodings.view(x.size(0), quantized.size(2), encodings.size(1))
        # encodings = encodings.view(x.size(0), quantized.size(2) * encodings.size(2))
        location = location_model(quantized)

        loss = F.mse_loss(location, torch.as_tensor(theta).float().to(device), reduction='sum')

        if (i + 1) % n_samples_test_on_validation_set == 0:  # Test on validation test for early stopping
            if len(val_error) > 2:
                print('previous_val_recon_error: %.3f' % last_error_val_test)
                if loss > last_error_val_test:
                    print('val_recon_error: %.3f' % loss.item())
                    # break
            last_error_val_test = loss

            print('val_recon_error: %.3f' % loss.item())
            val_error.append(loss.item())
            location_model.train()
        else:
            loss.backward()

            optimizer.step()

            train_location_error.append(loss.item())

        if (i + 1) % 10 == 0:
            print('%d iterations' % (i + 1))
            print('location error train: %.3f' % np.mean(train_location_error[-100:]))
            # print('location error test: %.3f' % np.mean(test_location_error[-100:]))
            print()

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.plot(train_location_error, label='train_dataset')
    ax.plot([(ind+1)*n_samples_test_on_validation_set for ind in range(len(val_error))], val_error)
    ax.legend()
    ax.set_yscale('log')
    ax.set_title('location estimation, Train vs Test error')
    ax.set_xlabel('iteration')

    torch.save(location_model, 'location_model.pt')
    plt.show()

if __name__ == '__main__':
    run_location_training()