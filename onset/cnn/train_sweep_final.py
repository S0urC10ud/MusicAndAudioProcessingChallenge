import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import wavfile
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import CustomCNN
import dill
from dataset import OnsetDetectionDataset
from dotdict import DotAccessibleDict
import librosa
from sklearn.metrics import confusion_matrix, accuracy_score
import mir_eval
import wandb
best_average_f1_score = 0

def print_histogram(data, bins=10, width=50):
    # Calculate histogram data
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)

    # Scale factor to fit the histogram in the given width
    scale = width / max_count

    # Print the histogram
    for count, edge in zip(hist, bin_edges[:-1]):
        bar = '#' * int(count * scale)
        print(f'{edge:10.3f} - {edge + (bin_edges[1] - bin_edges[0]):10.3f}: {bar}')


def train_test_split(dataset_dir, test_size=0.2, seed=42):
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.wav')]
    np.random.seed(seed)
    np.random.shuffle(files)
    split_idx = int(len(files) * (1 - test_size))
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    return train_files, test_files

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

def pick_delta_thresh(odf, threshold):
	peaks=[]
	for ind in range(1,len(odf)-1):
		if (odf[ind]>threshold) and (odf[ind+1] < odf[ind] > odf[ind-1]):
			peaks.append(ind)
	return np.array(peaks)

def process_file(file_path, model, config):
    def full_file_dataset(file_path, n_fft=[4096, 2048, 1024], hop_length=config.hop_length, n_mels=config.n_mels):
        sample_rate, audio = wavfile.read(file_path)
        if audio.dtype.kind == 'i':
            audio = audio.astype(float) / np.iinfo(audio.dtype).max
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        
        spectrograms = []
        #with open("expected_mus.npy", "rb") as expected_vals_handle:
        #    stored_mus = np.load(expected_vals_handle)
        #with open("expected_stds.npy", "rb") as expected_stds_handle:
        #    stored_stds = np.load(expected_stds_handle)
        for n_fft_option in n_fft:
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft_option, hop_length=config.hop_length, n_mels=config.n_mels, fmin=27.5, fmax=16000)
            log_spectrogram = 10*np.log10(1e-10+spectrogram)
            spectrograms.append(log_spectrogram)
        
        stacked_spectrograms = np.stack(spectrograms, axis=0)

        #for frequency_id in range(3):
        #    for mel_band in range(80):
        #        stacked_spectrograms[frequency_id][mel_band] = (stacked_spectrograms[frequency_id][mel_band] - stored_mus[frequency_id][mel_band]) / stored_stds[frequency_id][mel_band]

        return torch.tensor(stacked_spectrograms, dtype=torch.float32), sample_rate

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


    def load_correct_labels(onset_file, total_frames, frame_duration):
        if not os.path.exists(onset_file):
            return None
        with open(onset_file, 'r') as f:
            onsets = np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
        labels = np.zeros(total_frames, dtype=np.float32)
        frame_onsets = np.floor(onsets / frame_duration).astype(int)
        labels[frame_onsets] = 1.0
        return labels

    # Load the audio file and its labels
    audio_tensor, sample_rate = full_file_dataset(file_path)
    onset_labels = load_correct_labels(file_path.replace(".wav", ".onsets.gt"), audio_tensor.shape[2], config.hop_length / sample_rate)

    predictions = evaluate_model_on_full_file(model, audio_tensor, config.sample_delta)
    model.train()
    density_factor = (config.hop_length / sample_rate)
    return predictions, onset_labels, np.argwhere(onset_labels==1).flatten() * density_factor, density_factor

def post_process(detection_function, threshold, density_factor):
    #peak_indices = pick_peaks(predictions, threshold, change)
    peak_indices = pick_delta_thresh(detection_function, threshold)

    peak_times = peak_indices * density_factor
    return peak_indices, peak_times

def train_model(train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data, config, epochs=3500):
    global best_average_f1_score
    train_dataset = OnsetDetectionDataset(train_files, wavfile_train_data, onset_train_data, sample_delta=config.sample_delta, n_mels=config.n_mels, hop_length=config.hop_length)
    test_dataset = OnsetDetectionDataset(test_files, wavfile_test_data, onset_test_data, sample_delta=config.sample_delta, n_mels=config.n_mels, hop_length=config.hop_length, onset_one_in=8)

    # Use a higher number of workers and enable pin_memory for faster host to GPU transfers
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)

    model = CustomCNN(conv1_kernel_size=config.conv1_kernel_size, conv2_kernel_size=config.conv2_kernel_size, conv2_dimensions=config.conv2_dimensions, conv_out_dimensions=config.conv_out_dimensions, linear_out=config.linear_out,pool_size_1=config.pool_size_1, pool_size_2 =config.pool_size_2, dropout=config.dropout)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
        total_train_loss = 0
        model.train()
        all_train_preds, all_train_targets = [], []

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            weight = torch.where(targets == 1, torch.tensor(3.0), torch.tensor(1.0))
            criterion = nn.BCELoss(weight=weight)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # For accuracy and confusion matrix
            train_preds = (outputs > 0.5).to(dtype=int).squeeze()
            all_train_preds.extend(train_preds.cpu().numpy())
            all_train_targets.extend(targets.cpu().numpy())
        wandb.log({"epoch": epoch, "total_train_loss": total_train_loss})
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_targets, all_train_preds)
        train_confusion_mat = confusion_matrix(all_train_targets, all_train_preds)

        model.eval()

        # avg_test_loss = total_test_loss / len(test_loader)
        # test_accuracy = accuracy_score(all_test_targets, all_test_preds)
        # test_confusion_mat = confusion_matrix(all_test_targets, all_test_preds)
        # Save the model periodically or at certain epochs
        if (epoch+1) % 200 == 0:
            print("-"*40)
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')# Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            print("Train Confusion Matrix:\n", train_confusion_mat)
            with torch.no_grad():
                file_paths =['data/train_extra_onsets/ah_development_percussion_castagnet1.wav', 'data/train_extra_onsets/Media-105407(6.0-16.0).wav', 'data/train_extra_onsets/Media-104105(15.6-25.6).wav', 'data/train_extra_onsets/Media-106003(0.2-10.2).wav', 'data/train_extra_onsets/lame_velvet.wav', 'data/train_extra_onsets/ah_test_oud_Diverse_-_03_-_Muayyer_Kurdi_Taksim.wav', 'data/train_extra_onsets/Media-106001(9.7-19.7).wav', 'data/train_extra_onsets/ah_test_kemence_22_NevaTaksim_Kemence.wav', 'data/train_extra_onsets/SoundCheck2_60_Vocal_Tenor_opera.wav', 'data/train_extra_onsets/SoundCheck2_83_The_Alan_Parsons_Project_-_Limelight.wav', 'data/train_extra_onsets/ah_development_piano_MOON.wav', 'data/train_extra_onsets/lame_ftb_samp.wav', 'data/train_extra_onsets/Media-104306(5.0-15.0).wav', 'data/train_extra_onsets/train13.wav', 'data/train_extra_onsets/jpb_jaxx.wav', 'data/train_extra_onsets/gs_mix2_10dB.wav', 'data/train_extra_onsets/jpb_PianoDebussy.wav', 'data/train_extra_onsets/ff123_BigYellow.wav', 'data/train_extra_onsets/api_3-you_think_too_muchb.wav', 'data/train_extra_onsets/Media-105819(8.1-18.1).wav', 'data/train_extra_onsets/ff123_kraftwerk.wav', 'data/train_extra_onsets/vorbis_lalaw.wav', 'data/train_extra_onsets/ah_development_piano_autumn.wav', 'data/train_extra_onsets/Media-104111(5.0-15.0).wav', 'data/train_extra_onsets/ah_test_sax_Tubby_Hayes_-_The_Eighth_Wonder_-_11_-_Unidentified_12_Bar_Theme_pt1.wav', 'data/train_extra_onsets/ff123_Debussy.wav', 'data/train_extra_onsets/train8.wav', 'data/train_extra_onsets/ff123_BlueEyesExcerpt.wav']
                thresholds = [0.1, 0.25, 0.5, 0.7, 0.8, 0.85, 0.87, 0.9, 0.92, 0.95, 0.98]
                f1_scores = np.zeros((len(file_paths), len(thresholds)))
                for fid, file_path in enumerate(file_paths):
                    detection_function, onset_labels, onset_times, density_factor = process_file(file_path, model, config)
                    for tid, threshold in enumerate(thresholds):
                        peak_indices, predicted_onset_times = post_process(detection_function, threshold, density_factor)
                        f1_scores[fid, tid] = mir_eval.onset.f_measure(onset_times,
                                            predicted_onset_times,
                                            0.05)[0]
                        
            #print("Test Confusion Matrix:\n", test_confusion_mat)
            #print_histogram(all_test_preds, bins=5, width=10)
            print("-"*40)
            f1_score_sums = np.sum(f1_scores, axis=0)
            best_threshold_idx = np.argmax(f1_score_sums)
            curr_avg_f1_score = np.mean(f1_scores[:, best_threshold_idx])
            if curr_avg_f1_score > best_average_f1_score:
                print("New best model!")
            log_dict = {
                "epoch":epoch,
                "best_avg_f1_score": curr_avg_f1_score,
                "threshold": thresholds[best_threshold_idx],
                "avg_train_loss": avg_train_loss
            }
            print(log_dict)
            wandb.log(log_dict)

reload_data = True

def run_train(current_config):
    train_files, test_files = train_test_split('data/train_extra_onsets/', test_size=0.1, seed=42)
    wavfile_train_data = {}
    onset_train_data = {}

    if reload_data:
        for t in tqdm(train_files):
            sample_rate, audio = wavfile.read(t)
            if audio.dtype.kind == 'i':
                audio = audio.astype(float) / np.iinfo(audio.dtype).max
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            full_spectrograms = []
            for n_fft in [4096, 2048, 1024]:
                spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=current_config.hop_length, n_mels=current_config.n_mels, fmin=27.5, fmax=16000)
                log_spectrogram = 10*np.log10(1e-10+spectrogram)
                full_spectrograms.append(log_spectrogram)

            stacked_spectrograms = np.stack(full_spectrograms, axis=0)

            wavfile_train_data[t] = sample_rate, stacked_spectrograms

            with open(t.replace(".wav", ".onsets.gt"), 'r') as f:
                onset_train_data[t] =  np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
        
        # Code I used to generate pre-computed standardization values over the train set
        #expected_means = np.zeros(shape=(3, current_config.n_mels))
        #expected_stds = np.zeros(shape=(3, current_config.n_mels))
        #for frequency_id in range(3):
        #    for mel_band in range(current_config.n_mels):
        #        expected_means[frequency_id][mel_band] = np.average(np.array([np.average(v[1][frequency_id][mel_band]) for k,v in wavfile_train_data.items()]))
        #        expected_stds[frequency_id][mel_band] = np.average(np.array([np.std(v[1][frequency_id][mel_band]) for k,v in wavfile_train_data.items()]))
        #with open("expected_mus.npy", "wb") as expected_vals_handle:
        #    np.save(expected_vals_handle, expected_means)

        #with open("expected_stds.npy", "wb") as expected_stds_handle:
        #    np.save(expected_stds_handle, expected_stds)

        wavfile_test_data = {}
        onset_test_data = {}
        for t in tqdm(test_files):
            sample_rate, audio = wavfile.read(t)
            if audio.dtype.kind == 'i':
                audio = audio.astype(float) / np.iinfo(audio.dtype).max
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            full_spectrograms = []
            for n_fft in [4096, 2048, 1024]:
                spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=current_config.hop_length, n_mels=current_config.n_mels, fmin=27.5, fmax=16000)
                log_spectrogram = 10*np.log10(1e-10+spectrogram)
                full_spectrograms.append(log_spectrogram)

            stacked_spectrograms = np.stack(full_spectrograms, axis=0)

            wavfile_test_data[t] = sample_rate, stacked_spectrograms

            with open(t.replace(".wav", ".onsets.gt"), 'r') as f:
                onset_test_data[t] =  np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
        with open("onset_dataset.dill", "wb") as onset_dataset_file:
            dill.dump((train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data), onset_dataset_file)
    else:
        with open("onset_dataset.dill", "rb") as onset_dataset_file:
            train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data = dill.load(onset_dataset_file)
    

    wavfile_train_data.update(wavfile_test_data)
    onset_train_data.update(onset_test_data)
    train_model(train_files + test_files, wavfile_train_data, onset_train_data, [], {}, {}, current_config)

def run_sweep():
    with wandb.init() as run:
        current_config = DotAccessibleDict(run.config)
        run_train(current_config)

if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',
        'name': 'hyperparameter-sweep',
        'metric': {'name': 'best_avg_f1_score', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'values': [0.0001, 0.001, 0.005, 0.0005]},
            'batch_size': {'values': [16, 32, 64, 128, 256, 512]},
            'conv1_kernel_size': {'values': [(1,1),(3, 3), (2,2), (3, 7), (5, 5), (1, 4), (4, 12), (2, 3), (1, 4), (4,3), (3,7),(3,4), (2,1), (1,2)]},
            'conv2_kernel_size': {'values': [(1,1), (3, 3), (3, 7), (5, 5), (2, 2), (1, 1), (3, 2), (2, 3), (4, 2), (2, 4),(4,3),(3,4),(2,1), (1,2)]},
            'pool_size_1': {"values": [(1,1), (2,1), (3,1), (4,1), (1,2), (1,3), (1,4), (2,2), (2,3), (3,3), (3,4), (4,4)]},
            'pool_size_2': {"values": [(1,1), (2,1), (3,1), (4,1), (1,2), (1,3), (1,4), (2,2), (2,3), (3,3), (3,4), (4,4)]},
            "sample_delta": {"values": [7,14, 21, 5, 3,32]},
            'conv2_dimensions': {'values': [10, 20, 30, 50, 80, 100, 200, 400, 800]},
            'conv_out_dimensions': {'values': [10, 20, 40, 50, 100, 150, 200, 400, 600]},
            'linear_out': {'values': [8, 12, 16, 32, 64, 128, 150, 220]},
            'n_mels': {'values': [80]},
            'hop_length': {'values': [441*3,882, 661, 441, 220, 110]},
            "dropout": {"values": [0, 0.1, 0.01, 0.5, 0.3, 0.4, 0.2, 0.15]},
            "weight_decay":  {"values": [1e-3, 1e-4, 0, 1e-5, 1e-2]},
            "epochs": {"values": [5000]},
        }
    }
    #sweep_id = wandb.sweep(sweep_config, project="OnsetCNNFinalMoreHopSizes")
    wandb.agent("e6s13mgs", run_sweep, project="OnsetCNNFinalMoreHopSizes")