from .dotdict import DotAccessibleDict
import dill
import numpy as np
import librosa
import torch.nn.functional as F
import torch

cnn_config = DotAccessibleDict({
    'hop_length': 220,
    'conv1_kernel_size': (2,2),
    'conv2_kernel_size': (1,2),
    'conv2_dimensions': 50,
    'conv_out_dimensions': 20,
    'linear_out': 150,
    'n_mels': 80,
    'learning_rate': 0.0001,
    'batch_size': 32,
    "pool_size_1": [2,2],
    "pool_size_2": [3,1],
    "weight_decay": 0.,
    "dropout":0.01,
    "epochs": 10000,
    "sample_delta":14
})


def perform_inference(signal, sample_rate):
    with open("cnn_output.dmp", "rb") as dmp_file:
        model = dill.load(dmp_file)
        model.cuda()
    def prepare_data(signal, sample_rate, n_fft=[4096, 2048, 1024], hop_length=cnn_config.hop_length, n_mels=cnn_config.n_mels):
        spectrograms = []
        #with open("onset/cnn/expected_mus.npy", "rb") as expected_vals_handle:
        #    stored_mus = np.load(expected_vals_handle)
        #with open("onset/cnn/expected_stds.npy", "rb") as expected_stds_handle:
        #    stored_stds = np.load(expected_stds_handle)
        for n_fft_option in n_fft:
            spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft_option, hop_length=cnn_config.hop_length, n_mels=cnn_config.n_mels, fmin=27.5, fmax=16000)
            log_spectrogram = 10*np.log10(1e-10+spectrogram)
            spectrograms.append(log_spectrogram)
        
        stacked_spectrograms = np.stack(spectrograms, axis=0)

        #for frequency_id in range(3):
        #    for mel_band in range(80):
        #        stacked_spectrograms[frequency_id][mel_band] = (stacked_spectrograms[frequency_id][mel_band] - stored_mus[frequency_id][mel_band]) / stored_stds[frequency_id][mel_band]

        return torch.tensor(stacked_spectrograms, dtype=torch.float32)
    
    def evaluate_model_on_full_file(model, audio_tensor, step_offset):
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
    predictions = evaluate_model_on_full_file(model, audio_tensor, cnn_config.sample_delta)
    return predictions