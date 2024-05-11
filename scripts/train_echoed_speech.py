import torch
import os
from torch.utils.data import DataLoader
from six.moves import xrange
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from acoustic_locating_vq_vae.rir_dataset_generator.specsdataset import SpecsDataset
from acoustic_locating_vq_vae.visualization import plot_spectrogram
from acoustic_locating_vq_vae.data_preprocessing import spec_dataset_preprocessing
from acoustic_locating_vq_vae.vq_vae.echoed_speech_model import EchoedSpeechReconModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    rir_model = torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_rir.pt'))
    speech_model = torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_speech.pt'))

    BATCH_SIZE = 64
    num_training_updates = 15000
    num_hiddens = 80
    num_residual_layers = 2
    num_residual_hiddens = 80
    use_jitter = True
    LR = 1e-3
    n_samples_test_on_validation_set = 500

    DATASET_PATH = os.path.join(os.getcwd(), 'spec_data', '10k_set')
    VAL_DATASET_PATH = os.path.join(os.getcwd(), 'spec_data', 'val_set')

    train_data = SpecsDataset(DATASET_PATH)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda datum: spec_dataset_preprocessing(datum))

    last_error_val_test = float('inf')
    val_data = SpecsDataset(DATASET_PATH)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=lambda datum: spec_dataset_preprocessing(datum))

    sample_to_init, _, _, _, _, _ = next(iter(train_loader))
    out_channels = sample_to_init.shape[1]

    model = EchoedSpeechReconModel(rir_model, speech_model, out_channels, num_hiddens, num_residual_layers,
                                   num_residual_hiddens, use_jitter).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)

    train_res_recon_error = []
    train_speech_perp = []
    train_rir_perp = []
    val_error = []

    model.train()
    for i in xrange(num_training_updates):
        if (i + 1) % n_samples_test_on_validation_set == 0:  # Test on validation test for early stopping
            model.eval()
            _, _, echoed_specs, _, _, _ = next(iter(val_loader))
        else:
            _, _, echoed_specs, _, _, _ = next(iter(train_loader))
        x = echoed_specs.type(torch.FloatTensor)
        x = x.to(device)
        x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-8)

        x_rir = torch.permute(x, [0, 2, 1])

        optimizer.zero_grad()
        reconstructed_x, speech_perplexity, rir_perplexity = model(x, x_rir)

        if not x.shape == reconstructed_x.shape:
            reduction = reconstructed_x.shape[2] - x.shape[2]
            recon_error = F.mse_loss(reconstructed_x[:, :, :-reduction], x)
        else:
            recon_error = F.mse_loss(reconstructed_x, x)

        if (i + 1) % n_samples_test_on_validation_set == 0:  # Test on validation test for early stopping
            if len(val_error) > 2:
                print('previous_val_recon_error: %.3f' % last_error_val_test)
                if recon_error > last_error_val_test:
                    print('val_recon_error: %.3f' % recon_error.item())
                    # break
            last_error_val_test = recon_error

            print('val_recon_error: %.3f' % recon_error.item())
            val_error.append(recon_error.item())
            model.train()
        else:
            loss = recon_error
            loss.backward()

            optimizer.step()

            train_res_recon_error.append(loss.item())
            train_speech_perp.append(speech_perplexity.item())
            train_rir_perp.append(rir_perplexity.item())

        if (i + 1) % 10 == 0:
            print('==========================================')
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('speech perplexity: %.3f' % np.mean(train_speech_perp[-100:]))
            print('rir perplexity: %.3f' % np.mean(train_rir_perp[-100:]))
        if (i + 1) % 500 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plot_spectrogram(torch.squeeze(x[0]).detach().to('cpu'), title="Spectrogram - input", ylabel="mag", ax=ax1)
            plot_spectrogram(torch.squeeze(reconstructed_x[0]).detach().to('cpu'), title="Spectrogram - reconstructed",
                             ylabel="mag",
                             ax=ax2)
            plt.show()

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 1, 1)
    ax.plot(train_res_recon_error, label='train_dataset')
    ax.plot([(ind+1)*n_samples_test_on_validation_set for ind in range(len(val_error))], val_error,
            label='validation_dataset')
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    plt.show()


    torch.save(model, '../models/model_echoed_speech.pt')
