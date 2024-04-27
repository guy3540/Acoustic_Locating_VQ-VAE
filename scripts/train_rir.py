import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from six.moves import xrange

from acustic_locating_vq_vae.data_preprocessing import combine_tensors_with_min_dim
from acustic_locating_vq_vae.rir_dataset_generator.rir_dataset import RIR_DATASET
from acustic_locating_vq_vae.visualization import plot_spectrogram

from acustic_locating_vq_vae.vq_vae.convolutional_vq_vae import ConvolutionalVQVAE
from acustic_locating_vq_vae.vq_vae.location_model.location_model import LocationModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rir_data_preprocessing(data):
    spectrograms = []
    source_coordinates_list = []
    mic_list = []
    room_list = []
    fs_list = []
    for (spec, source_coordinates, mic, room, fs) in data:
        spectrograms.append(torch.unsqueeze(torch.from_numpy(spec), dim=0))
        source_coordinates_list.append(source_coordinates)
        mic_list.append(mic)
        room_list.append(room)
        fs_list.append(fs)
    spectrograms = combine_tensors_with_min_dim(spectrograms)

    return spectrograms, source_coordinates_list, mic_list, room_list, fs_list


def train_vq_vae(model: ConvolutionalVQVAE, optimizer, train_loader, num_training_updates):
    model.train()

    train_res_recon_error = []
    train_res_perplexity = []

    # waveform B,C,S
    for i in xrange(num_training_updates):
        x, source_coordinates, mic, room, fs = next(iter(train_loader))
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = torch.unsqueeze(x, 1)

        optimizer.zero_grad()
        vq_loss, reconstructed_x, perplexity = model(x)

        if not x.shape == reconstructed_x.shape:
            reduction = reconstructed_x.shape[2] - x.shape[2]
            recon_error = F.mse_loss(reconstructed_x[:, :, :-reduction], x)  # / data_variance
        else:
            recon_error = F.mse_loss(reconstructed_x, x)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
        if (i + 1) % 500 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plot_spectrogram(torch.squeeze(x[0]).detach().to('cpu'), title="Spectrogram - input", ylabel="mag", ax=ax1)
            plot_spectrogram(torch.squeeze(reconstructed_x[0]).detach().to('cpu'), title="Spectrogram - reconstructed",
                             ylabel="mag",
                             ax=ax2)
            plt.show()

    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')
    plt.show()
    torch.save(model, 'model_rir.pt')


def train_location(vae_model: ConvolutionalVQVAE, location_model, optimizer, num_training_updates, train_loader):
    vae_model.eval()
    location_model.train()

    train_location_error = []
    test_location_error = []

    # waveform B,C,S
    for i in xrange(num_training_updates):

        x, source_coordinates, mic, room, fs = next(iter(train_loader))
        source_coordinates = source_coordinates.to(device)
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = torch.unsqueeze(x, 1)

        optimizer.zero_grad()
        loss, quantized, perplexity, encodings = vae_model.get_latent_representation(x)
        encodings = encodings.view(x.size(0), quantized.size(2), encodings.size(1))
        encodings = encodings.view(x.size(0), quantized.size(2) * encodings.size(2))
        location = location_model(encodings)

        loss = F.mse_loss(location, torch.squeeze(source_coordinates).float())
        loss.backward()

        optimizer.step()

        train_location_error.append(loss.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('location error train: %.3f' % np.mean(train_location_error[-100:]))
            mea_test_error = evaluate_model(location_model)
            location_model.train()
            print('location error test: %.3f' % mea_test_error)
            print()
            test_location_error.append(mea_test_error)

    train_res_recon_error_smooth = savgol_filter(train_location_error, 201, 7)
    test_location_error_smooth = savgol_filter(test_location_error, 201, 7)

    f = plt.figure()
    ax = f.add_subplot(2, 1, 1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed MSE. Train')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(2, 1, 2)
    ax.plot(test_location_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed MSE. TEST')
    ax.set_xlabel('iteration/100')

    torch.save(location_model, 'location_model.pt')
    plt.show()


def run_location_training():
    DATASET_PATH = Path(os.getcwd()) / 'rir_dataset_generator' / 'dev_data'
    encoder_output_dim = 101
    num_embeddings = 512
    BATCH_SIZE = 64
    num_training_updates = 15000

    train_data = RIR_DATASET(DATASET_PATH)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    vae_model = torch.load('model_rir.pt').to(device)
    location_model = LocationModule(encoder_output_dim, num_embeddings, 3).to(device)
    optimizer = torch.optim.Adam(location_model.parameters(), lr=1e-3)

    train_location(vae_model, location_model, optimizer, num_training_updates, train_loader)


def run_rir_training():
    DATASET_PATH = Path(os.getcwd()) / 'rir_dataset_generator' / 'dev_data'
    BATCH_SIZE = 64
    LR = 1e-3
    # SAMPLING_RATE = 16e3
    # NFFT = int(SAMPLING_RATE * 0.025)
    IN_FEATURE_SIZE = 1
    # HOP_LENGTH = int(SAMPLING_RATE * 0.01)

    # CONV VQVAE
    output_features_dim = IN_FEATURE_SIZE
    num_hiddens = 40
    in_channels = IN_FEATURE_SIZE
    num_residual_layers = 2
    num_residual_hiddens = 20
    embedding_dim = 40
    num_embeddings = 512  # The higher this value, the higher the capacity in the information bottleneck.
    commitment_cost = 0.25  # as recommended in VQ VAE article
    use_jitter = False
    jitter_probability = 0.12

    train_data = RIR_DATASET(DATASET_PATH)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = ConvolutionalVQVAE(in_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens,
                               commitment_cost, num_embeddings, use_jitter=use_jitter).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
    train_vq_vae(model=model, optimizer=optimizer, train_loader=train_loader, num_training_updates=15000)


def evaluate_model(location_model=torch.load('location_model.pt').to(device)):
    DATASET_PATH = Path(os.getcwd()) / 'rir_dataset_generator' / 'eval_data'
    test_data = RIR_DATASET(DATASET_PATH)
    BATCH_SIZE = 1
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    vae_model = torch.load('model_rir.pt').to(device)
    # location_model = torch.load('location_model.pt').to(device)

    vae_model.eval()
    location_model.eval()
    loss_list = []
    for i in range(100):
        x, source_coordinates, mic, room, fs = next(iter(test_loader))
        source_coordinates = torch.squeeze(source_coordinates).to(device)
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
        x = torch.unsqueeze(x, 1)

        loss, quantized, perplexity, encodings = vae_model.get_latent_representation(x)
        location = location_model(torch.flatten(encodings))
        loss = F.mse_loss(location, source_coordinates.float())
        loss_list.append(loss.item())
        # print(loss_list[-1])
        # print(
        #     f"original location: {source_coordinates[0].item():.2f},{source_coordinates[1].item():.2f},{source_coordinates[2].item():.2f},"
        #     f" estimated location: {location[0].item():.2f},{location[1].item():.2f},{location[2].item():.2f}")
    mean_loss = torch.mean(torch.as_tensor(loss_list))
    return mean_loss


if __name__ == '__main__':
    # run_rir_training()
    run_location_training()
    # evaluate_model()
