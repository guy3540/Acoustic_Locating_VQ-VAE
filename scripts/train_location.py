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
    DATASET_PATH = Path(os.getcwd()) / 'spec_data' / '20k_set'
    VAL_DATASET_PATH = Path(os.getcwd()) / 'spec_data' / 'val_set'
    encoder_output_dim = 201
    embedding_dim = 1024
    BATCH_SIZE = 16
    num_training_updates = 15000
    train_percent = 0.95

    train_data = SpecsDataset(DATASET_PATH)
    val_data = SpecsDataset(VAL_DATASET_PATH)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: spec_dataset_preprocessing(x))

    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: spec_dataset_preprocessing(x))

    echoed_speech_model = torch.load("/home/guy/PycharmProjects/Acoustic_Locating_VQ-VAE/models/model_echoed_speech_15000.pt").to(device)
    location_model = LocationModule(encoder_output_dim, embedding_dim, 1).to(device)
    optimizer = torch.optim.Adam(location_model.parameters(), lr=1e-3)

    train_location(echoed_speech_model, location_model, optimizer, num_training_updates, train_loader, val_loader)


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
        x_trans = x.permute(0, 2, 1)

        optimizer.zero_grad()
        _, quantized, perplexity, encodings = combined_model.rir_model.get_latent_representation(x_trans)

        _, quantized_s, perplexity_s, encodings_s = combined_model.speech_model.get_latent_representation(x)
        # encodings = encodings.view(x.size(0), quantized.size(2), encodings.size(1))
        # encodings = encodings.view(x.size(0), quantized.size(2) * encodings.size(2))
        encodings = encodings.reshape(quantized.shape[0], 201, encodings.shape[1])
        location = location_model(encodings)

        loss = F.mse_loss(location, torch.as_tensor(theta).float().to(device)/torch.pi,
                          reduction='mean')

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
        if (i + 1) % 50 == 0:
            f, (ax1, ax2) = plt.subplots(1, 2)

            ax1.plot(location.cpu().detach())
            ax1.plot(theta.cpu().detach() / torch.pi)

            ax2.plot(train_location_error, label='train_dataset')
            ax2.plot([(ind + 1) * n_samples_test_on_validation_set for ind in range(len(val_error))], val_error,
                    label='validation_dataset')
            ax2.legend()
            ax2.set_yscale('log')
            ax2.set_title('location estimation, Train vs Test error')
            ax2.set_xlabel('iteration')
            plt.show()
        if (i + 1) % 1000 == 0:
            torch.save(location_model, '../models/model_location_' + str(i + 1) + '.pt')

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.plot(train_location_error, label='train_dataset')
    ax.plot([(ind+1)*n_samples_test_on_validation_set for ind in range(len(val_error))], val_error,
            label='validation_dataset')
    ax.legend()
    ax.set_yscale('log')
    ax.set_title('location estimation, Train vs Test error')
    ax.set_xlabel('iteration')

    torch.save(location_model, '../models/location_model_final.pt')
    plt.show()

if __name__ == '__main__':
    run_location_training()