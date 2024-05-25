from .dotdict import DotAccessibleDict
import dill
import numpy as np
import librosa
import torch.nn.functional as F
import torch


cnn_config = DotAccessibleDict({
    'hop_length': 44,
    'conv1_kernel_size': (1,4),
    'conv2_kernel_size': (3,4),
    'conv2_dimensions': 100,
    'conv_out_dimensions': 600,
    'linear_out': 64,
    'n_mels': 80,
    'learning_rate': 0.001,
    'batch_size': 64,
    "pool_size_1": [3,3],
    "pool_size_2": [2,1],
    "weight_decay": 0.001
})

def pick_peaks(data, threshold, min_distance):
    peaks = []
    potential_peak = None
    for i in range(len(data)):
        if data[i] > threshold:
            if potential_peak is None or data[i] > data[potential_peak]:
                potential_peak = i
        if potential_peak is not None and i >= potential_peak + min_distance:
            peaks.append(potential_peak)
            potential_peak = None
    if potential_peak is not None:
        peaks.append(potential_peak)
    return np.array(peaks)

def perform_inference(signal, sample_rate):
    with open("onset/cnn/cnn_output_well_trained.dmp", "rb") as dmp_file:
        model = dill.load(dmp_file)
        model.cuda()
    def prepare_data(signal, sample_rate, n_fft=[4096, 2048, 1024], hop_length=cnn_config.hop_length, n_mels=cnn_config.n_mels):
        spectrograms = []
        with open("onset/cnn/expected_mus.npy", "rb") as expected_vals_handle:
            stored_mus = np.load(expected_vals_handle)
        with open("onset/cnn/expected_stds.npy", "rb") as expected_stds_handle:
            stored_stds = np.load(expected_stds_handle)
        for n_fft_option in n_fft:
            spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft_option, hop_length=hop_length, n_mels=n_mels)
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            spectrograms.append(log_spectrogram)
        
        stacked_spectrograms = np.stack(spectrograms, axis=0)

        for frequency_id in range(3):
            for mel_band in range(80):
                stacked_spectrograms[frequency_id][mel_band] = (stacked_spectrograms[frequency_id][mel_band] - stored_mus[frequency_id][mel_band]) / stored_stds[frequency_id][mel_band]

        return torch.tensor(stacked_spectrograms, dtype=torch.float32)
    
    def evaluate_model_on_full_file(model, audio_tensor, step_offset=7):
        model.eval()
        num_chunks = audio_tensor.shape[2]
        predictions = np.zeros(num_chunks)
        with torch.no_grad():
            # Process audio in chunks of size `sample_length`
            for i in range(step_offset, num_chunks - step_offset - 1):
                chunk = audio_tensor[:, :, i-step_offset : i + step_offset + 1]
                output = model(chunk.unsqueeze(0).cuda())
                predicted_prob = output.cpu().numpy()
                predictions[i] = predicted_prob
        return np.array(predictions)
    
    audio_tensor = prepare_data(signal, sample_rate)
    predictions = evaluate_model_on_full_file(model, audio_tensor)
    return predictions