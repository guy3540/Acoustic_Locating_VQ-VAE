import torch
import numpy as np
import scipy.signal as ss

def speech_waveform_to_spec(waveform, sample_rate, NFFT, noverlap):
    f, t, spec = ss.stft(waveform.squeeze(), nperseg=NFFT, noverlap=noverlap, fs=sample_rate)
    a = np.real(spec)
    b = np.imag(spec)
    spec_final = np.vstack((np.real(spec), np.imag(spec)))

    return spec_final


def batchify_spectrograms(data, NFFT, noverlap):
    spectrograms = []
    for (waveform, _, _, _, _, sample_rate) in data:
        R_ri = waveform
        spectrograms.append(R_ri.unsqueeze(0))

    spectrograms = combine_tensors_with_min_dim(spectrograms)

    return spectrograms, sample_rate,  # transcript, speaker_id, chapter_id, utterance_id


def rir_data_preprocessing(data):
    spectrograms = []
    winner_est_list = []
    source_coordinates_list = []
    mic_list = []
    room_list = []
    fs_list = []
    for (spec, winner_est, source_coordinates, mic, room, fs) in data:
        if spec.shape[1] < 500:
            continue
        else:
            ispec = spec[:, :500]
        spectrograms.append(torch.unsqueeze(torch.from_numpy(ispec), dim=0))
        source_coordinates_list.append(source_coordinates)
        mic_list.append(mic)
        room_list.append(room)
        fs_list.append(fs)
        winner_est_list.append(winner_est)
    spectrograms = combine_tensors_with_min_dim(spectrograms)

    return spectrograms, torch.as_tensor(
        np.asarray(winner_est_list)), source_coordinates_list, mic_list, room_list, fs_list


def rir_data_preprocess_permute_normalize_and_cut(data, max_size: int = 500):
    spectrograms = []
    winner_est_list = []
    source_coordinates_list = []
    mic_list = []
    room_list = []
    fs_list = []
    for (spec, winner_est, source_coordinates, mic, room, fs) in zip(*data):
        if spec.shape[1] < max_size:
            continue
        else:
            ispec = spec[:, :max_size]
            ispec = (ispec - torch.mean(ispec, dim=1, keepdim=True)) / (torch.std(ispec, dim=1, keepdim=True) + 1e-8)
            ispec = torch.permute(ispec, [1, 0])
            ispec = ispec.type(torch.FloatTensor)
        spectrograms.append(torch.unsqueeze(ispec, dim=0))
        source_coordinates_list.append(source_coordinates)
        mic_list.append(mic)
        room_list.append(room)
        fs_list.append(fs)
        winner_est = winner_est.type(torch.FloatTensor)
        winner_est = (winner_est - torch.mean(winner_est)) / (torch.std(winner_est) + 1e-8)
        winner_est = torch.unsqueeze(winner_est, 0)
        winner_est_list.append(winner_est)
    spectrograms = combine_tensors_with_min_dim(spectrograms)

    return spectrograms, torch.stack(winner_est_list, 0), source_coordinates_list, mic_list, room_list, fs_list


def combine_tensors_with_min_dim(tensor_list):
    """
  Combines a list of PyTorch tensors with shapes (1, H, x1), (1, H, x2), ..., (1, H, xN)
  into a new tensor of shape (N, H, X), where X is the minimum dimension among x1, x2, ..., xN.

  Args:
      tensor_list: A list of PyTorch tensors with the same height (H).

  Returns:
      A new tensor of shape (N, H, X), where X is the minimum dimension.

  Raises:
      ValueError: If the tensors in the list do not have the same height (H).
  """

    if not tensor_list:
        raise ValueError("Input tensor list cannot be empty")

    # Check if all tensors have the same height (H)
    H = tensor_list[0].size(1)
    for tensor in tensor_list:
        if tensor.size(1) != H:
            raise ValueError("All tensors in the list must have the same height (H)")

    # Get the minimum dimension (X) across all tensors in the list
    min_dim = min(tensor.size(2) for tensor in tensor_list)

    # Create a new tensor to store the combined data
    combined_tensor = torch.zeros((len(tensor_list), H, min_dim))

    # Fill the combined tensor with data from the input tensors, selecting the minimum value for each element
    for i, tensor in enumerate(tensor_list):
        combined_tensor[i, :, :] = tensor[:, :, :min_dim]

    return combined_tensor
