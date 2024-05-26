import torch
import numpy as np
import scipy.signal as ss
import rir_generator as rir


def batchify_spectrograms(data, NFFT, noverlap):
    spectrograms = []
    for (waveform, _, _, _, _, sample_rate) in data:
        R_ri = waveform
        spectrograms.append(R_ri.unsqueeze(0))

    spectrograms = combine_tensors_with_min_dim(spectrograms)

    return spectrograms, sample_rate,  # transcript, speaker_id, chapter_id, utterance_id


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
    H = tensor_list[0].shape[1]
    for tensor in tensor_list:
        if tensor.shape[1] != H:
            raise ValueError("All tensors in the list must have the same height (H)")

    # Get the minimum dimension (X) across all tensors in the list
    min_dim = min(tensor.shape[2] for tensor in tensor_list)

    # Create a new tensor to store the combined data
    combined_tensor = torch.zeros((len(tensor_list), H, min_dim), dtype=torch.complex64)

    # Fill the combined tensor with data from the input tensors, selecting the minimum value for each element
    for i, tensor in enumerate(tensor_list):
        combined_tensor[i, :, :] = tensor[:, :, :min_dim]

    return combined_tensor


def spec_dataset_preprocessing(data):
    speech_spec_list = []
    rir_spec_list = []
    echoed_spec_list = []
    wiener_est_list = []
    theta_est_list = []
    fs_list = []

    for (speech_spec, rir_spec, echoed_spec, sample_rate, theta, wiener_est) in data:
        if speech_spec.shape[1] < 500:
            continue
        else:
            speech_spec = speech_spec[:, :500]
            rir_spec = rir_spec[:, :500]
            echoed_spec = echoed_spec[:, :500]

        speech_spec_list.append(speech_spec)
        rir_spec_list.append(rir_spec)
        echoed_spec_list.append(echoed_spec)

        wiener_est_list.append(wiener_est)
        theta_est_list.append(theta)
        fs_list.append(torch.as_tensor(sample_rate))

    if len(speech_spec_list) == 0:
        speech_specs= rir_specs= echoed_specs= fs_tensor= theta_tensor= wiener_est_tensor= []
        return speech_specs, rir_specs, echoed_specs, fs_tensor, theta_tensor, wiener_est_tensor
    speech_specs = torch.stack(speech_spec_list)
    rir_specs = torch.stack(rir_spec_list)
    echoed_specs = torch.stack(echoed_spec_list)
    fs_tensor = torch.stack(fs_list)
    theta_tensor = torch.stack(theta_est_list)
    wiener_est_tensor = torch.stack(wiener_est_list)

    return speech_specs, rir_specs, echoed_specs, fs_tensor, theta_tensor, wiener_est_tensor
