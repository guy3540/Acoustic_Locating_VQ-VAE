import torch
import numpy as np
import scipy.signal as ss
import rir_generator as rir


def get_real_spec_from_complex(spec: torch.tensor):
    real = torch.real(spec)
    imag = torch.imag(spec)
    return torch.cat((real, imag), dim=1)


def get_complex_spec_from_real(spec: torch.Tensor):
    real_part = spec[:, :spec.shape[1] // 2]
    imag_part = spec[:, spec.shape[1] // 2:]
    return torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))

def speech_waveform_to_spec(waveform, sample_rate, NFFT, noverlap):
    if waveform.shape[2] < sample_rate*3:
        return None
    else:
        waveform = waveform[:, :, :sample_rate*3]
        waveform = (waveform - waveform.mean()) / waveform.std()
    f, t, spec = ss.stft(waveform.squeeze(), nperseg=NFFT, noverlap=noverlap, fs=sample_rate)
    return spec


def batchify_spectrograms(data, NFFT, noverlap):
    spectrograms = []
    for (waveform, _, _, _, _, sample_rate) in data:

        R_ri = waveform
        spectrograms.append(R_ri.unsqueeze(0))

    spectrograms = combine_tensors_with_min_dim(spectrograms)

    return spectrograms, sample_rate,  # transcript, speaker_id, chapter_id, utterance_id

def batchify_echoed_speech(data):
    echoed_spectrograms_list = []
    rir_spectrograms_list = []
    unechoed_spectrograms_list = []
    wiener_est_list = []
    theta_list = []

    for (echoed_speech_spec, rir_spec, unechoed_spec, sample_rate, theta, wiener_est) in data:
        echoed_spectrograms_list.append(echoed_speech_spec)
        rir_spectrograms_list.append(rir_spec)
        unechoed_spectrograms_list.append(unechoed_spec)
        wiener_est_list.append(wiener_est.unsqueeze(2))
        theta_list.append(theta)

    echoed_spectrograms = combine_tensors_with_min_dim(echoed_spectrograms_list)
    rir_spectrograms = combine_tensors_with_min_dim(rir_spectrograms_list)
    unechoed_spectrograms = combine_tensors_with_min_dim(unechoed_spectrograms_list)
    wiener_est_final = combine_tensors_with_min_dim(wiener_est_list)

    return echoed_spectrograms, rir_spectrograms, unechoed_spectrograms, sample_rate, theta_list, wiener_est_final


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


def echoed_spec_from_random_rir(data, Z_LOC_SOURCE, R, room_dimensions, receiver_position, fs, reverberation_time,
                                n_sample, C, NFFT, noverlap, **kwargs):
    theta = np.random.uniform(low=-np.pi, high=np.pi, size=1)
    z_loc = np.array([Z_LOC_SOURCE])
    h_src_loc = np.stack((R * np.cos(theta).T, R * np.sin(theta).T, z_loc), axis=1) + receiver_position
    h_src_loc = np.minimum(h_src_loc, room_dimensions)
    h_RIR = rir.generate(
        c=C,  # Sound velocity (m/s)
        fs=int(fs),  # Sample frequency (samples/s)
        r=receiver_position,
        s=np.squeeze(h_src_loc),  # Source position [x y z] (m)
        L=room_dimensions,  # Room dimensions [x y z] (m)
        reverberation_time=reverberation_time,  # Reverberation time (s)
        nsample=n_sample,  # Number of output samples
    )

    echoed_spec_list = []
    rir_spec_list = []
    unechoed_spec_list = []
    theta_list = []
    wiener_est_list = []
    sample_rate_list = []

    for (waveform, sample_rate, _, _, _, _) in data:
        spec_signal = speech_waveform_to_spec(waveform.unsqueeze(0), sample_rate, NFFT, noverlap)
        if spec_signal is None:
            continue
        waveform_h = ss.convolve(waveform.squeeze(), h_RIR.squeeze(), mode='same')
        spec_with_h = speech_waveform_to_spec(np.expand_dims(waveform_h, axis=(0, 1)), sample_rate, NFFT, noverlap)

        rir_spec = np.divide(spec_with_h, spec_signal + 1e-8)

        wiener_est = np.sum(spec_with_h * np.conjugate(spec_signal), axis=1) / (
                    np.sum(spec_signal * np.conjugate(spec_signal), axis=1) + 1e-8)

        echoed_spec_list.append(torch.from_numpy(spec_with_h).unsqueeze(0))
        rir_spec_list.append(torch.from_numpy(rir_spec).unsqueeze(0))
        unechoed_spec_list.append(torch.from_numpy(spec_signal).unsqueeze(0))
        theta_list.append(torch.from_numpy(theta).unsqueeze(0))
        wiener_est_list.append(torch.from_numpy(wiener_est).unsqueeze(0))
        sample_rate_list.append(sample_rate)

    return echoed_spec_list, rir_spec_list, unechoed_spec_list, sample_rate_list, theta_list, wiener_est_list