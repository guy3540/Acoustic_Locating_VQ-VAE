import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from acoustic_locating_vq_vae.data_preprocessing import spec_dataset_preprocessing
from acoustic_locating_vq_vae.rir_dataset_generator.specsdataset import SpecsDataset

from sklearn.manifold import TSNE
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    fixed_rir_dataset = Path(os.getcwd()) / 'spec_data' / 'Fixed_rir'
    fixed_speech_dataset = Path(os.getcwd()) / 'spec_data' / 'Fixed_speech'
    one_k_dataset = Path(os.getcwd()) / 'spec_data' / '1k_samples'

    model = torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_echoed_speech_6500.pt'))
    model.eval()

    BATCH_SIZE = 1

    # fixed_rir_data = SpecsDataset(fixed_rir_dataset)
    # fixed_speech_data = SpecsDataset(fixed_speech_dataset)
    one_k_data = SpecsDataset(one_k_dataset)
    # fixed_rir_loader = DataLoader(fixed_rir_data, batch_size=BATCH_SIZE, shuffle=True,
    #                          collate_fn=lambda datum: spec_dataset_preprocessing(datum))
    # fixed_speech_loader = DataLoader(fixed_speech_data, batch_size=BATCH_SIZE, shuffle=True,
    #                          collate_fn=lambda datum: spec_dataset_preprocessing(datum))
    one_k_loader = DataLoader(one_k_data, batch_size=BATCH_SIZE, shuffle=True,
                             collate_fn=lambda datum: spec_dataset_preprocessing(datum))

    speech_quantized_list = []
    rir_quantized_list = []
    theta_list = []

    for i in range(len(one_k_data)):
        speech_spec, rir_spec, echoed_spec, sample_rate, theta, wiener_est = next(iter(one_k_loader))
        if len(speech_spec) == 0:
            i-=1
            continue
        elif speech_spec.shape[2] <500:
            continue
        with torch.no_grad():
            theta_list.append(theta)

            x = echoed_spec.type(torch.FloatTensor)
            x = x.to(device)
            x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)
            x_trans = x.permute(0, 2, 1)

            _, quantized, _, encodings = model.rir_model.get_latent_representation(x_trans)
            rir_quantized_list.append(encodings.flatten().cpu())

            _, quantized, _, encodings = model.speech_model.get_latent_representation(x)
            speech_quantized_list.append(encodings.flatten().cpu())

    rir = torch.stack(rir_quantized_list, dim=0)
    speech = torch.stack(speech_quantized_list, dim=0)
    theta = torch.stack(theta_list, dim=0)

    emb = TSNE(n_components=2, perplexity=100, n_iter=1000).fit_transform(rir)

    fig, ax = plt.subplots()
    s = ax.scatter(emb[:, 0], emb[:, 1], c=theta)
    plt.colorbar(s)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(emb[0:(a.shape[0] - 1), 0], emb[0:(a.shape[0] - 1), 1])
    ax.scatter(emb[a.shape[0]:, 0], emb[a.shape[0]:, 1])
    ax.set_title("Fixed RIR & Fixed Speech")
    plt.show()

    print("Done")



