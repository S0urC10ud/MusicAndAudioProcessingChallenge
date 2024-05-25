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
import wandb
from dataset import OnsetDetectionDataset
from dotdict import DotAccessibleDict
import librosa
from sklearn.metrics import confusion_matrix, accuracy_score

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

def train_model(train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data, config, epochs=2500):
    train_dataset = OnsetDetectionDataset(train_files, wavfile_train_data, onset_train_data, sample_delta=7, n_mels=config.n_mels, hop_length=config.hop_length)
    test_dataset = OnsetDetectionDataset(test_files, wavfile_test_data, onset_test_data, sample_delta=7, n_mels=config.n_mels, hop_length=config.hop_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)

    model = CustomCNN(conv1_kernel_size=config.conv1_kernel_size, conv2_kernel_size=config.conv2_kernel_size, conv2_dimensions=config.conv2_dimensions, conv_out_dimensions=config.conv_out_dimensions, linear_out=config.linear_out,pool_size_1=config.pool_size_1, pool_size_2 =config.pool_size_2, dropout=config.dropout)
    model.cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        all_train_preds, all_train_targets = [], []

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            train_preds = (outputs > 0.5).to(dtype=int).squeeze()
            all_train_preds.extend(train_preds.cpu().numpy())
            all_train_targets.extend(targets.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_targets, all_train_preds)
        train_confusion_mat = confusion_matrix(all_train_targets, all_train_preds)

        model.eval()
        total_test_loss = 0
        all_test_preds, all_test_targets = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets.reshape(-1, 1))
                total_test_loss += loss.item()

                test_preds = (outputs > 0.5).to(dtype=int).squeeze()
                all_test_preds.extend(test_preds.cpu().numpy())
                all_test_targets.extend(targets.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = accuracy_score(all_test_targets, all_test_preds)
        test_confusion_mat = confusion_matrix(all_test_targets, all_test_preds)
        
        wandb.log({
            'epoch': epoch,
            'avg_train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'avg_test_loss': avg_test_loss,
            'test_accuracy': test_accuracy
        })

        if (epoch + 1) % 25 == 0:
            print("-" * 40)
            print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            print("Train Confusion Matrix:\n", train_confusion_mat)
            print("Test Confusion Matrix:\n", test_confusion_mat)
            print_histogram(all_test_preds, bins=5, width=10)
            print("-" * 40)
            with open(f"cnn_output.dmp", "wb") as dmp_file:
                dill.dump(model, dmp_file)

current_config = DotAccessibleDict({
    'hop_length': 44,
    'conv1_kernel_size': (3, 7),
    'conv2_kernel_size': (3, 3),
    'conv2_dimensions': 20,
    'conv_out_dimensions': 50,
    'linear_out': 256,
    'n_mels': 80,
    'learning_rate': 0.00001,
    'batch_size': 40
})

reload_data = False

def sweep_train():
    with wandb.init() as run:
        config = DotAccessibleDict(run.config)
        train_files, test_files = train_test_split('data/train_extra_onsets/', test_size=0.2, seed=42)
        wavfile_train_data = {}
        onset_train_data = {}
        
        if reload_data:
            with open("expected_mus.npy", "rb") as expected_vals_handle:
                stored_mus = np.load(expected_vals_handle)
            with open("expected_stds.npy", "rb") as expected_stds_handle:
                stored_stds = np.load(expected_stds_handle)

            for t in tqdm(train_files):
                sample_rate, audio = wavfile.read(t)
                if audio.dtype.kind == 'i':
                    audio = audio.astype(float) / np.iinfo(audio.dtype).max
                if audio.ndim == 2:
                    audio = np.mean(audio, axis=1)

                full_spectrograms = []
                for n_fft in [4096, 2048, 1024]:
                    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=current_config.hop_length, n_mels=current_config.n_mels)
                    log_spectrogram = librosa.power_to_db(spectrogram)
                    full_spectrograms.append(log_spectrogram)

                stacked_spectrograms = np.stack(full_spectrograms, axis=0)

                for frequency_id in range(3):
                    for mel_band in range(current_config.n_mels):
                        stacked_spectrograms[frequency_id][mel_band] = (stacked_spectrograms[frequency_id][mel_band] - stored_mus[frequency_id][mel_band]) / stored_stds[frequency_id][mel_band]

                wavfile_train_data[t] = sample_rate, stacked_spectrograms

                with open(t.replace(".wav", ".onsets.gt"), 'r') as f:
                    onset_train_data[t] = np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
            
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
                    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=current_config.hop_length, n_mels=current_config.n_mels)
                    log_spectrogram = librosa.power_to_db(spectrogram)
                    full_spectrograms.append(log_spectrogram)

                stacked_spectrograms = np.stack(full_spectrograms, axis=0)

                for frequency_id in range(3):
                    for mel_band in range(current_config.n_mels):
                        stacked_spectrograms[frequency_id][mel_band] = (stacked_spectrograms[frequency_id][mel_band] - stored_mus[frequency_id][mel_band]) / stored_stds[frequency_id][mel_band]

                wavfile_test_data[t] = sample_rate, stacked_spectrograms

                with open(t.replace(".wav", ".onsets.gt"), 'r') as f:
                    onset_test_data[t] = np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
            with open("onset_dataset.dill", "wb") as onset_dataset_file:
                dill.dump((train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data), onset_dataset_file)
        else:
            with open("onset_dataset.dill", "rb") as onset_dataset_file:
                train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data = dill.load(onset_dataset_file)
            
        train_model(train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data, config)

if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',
        'name': 'hyperparameter-sweep',
        'metric': {'name': 'avg_test_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'values': [0.0001, 0.001, 0.005, 0.0005]},
            'batch_size': {'values': [16, 32, 64, 128, 256, 512]},
            'conv1_kernel_size': {'values': [(3, 3), (3, 7), (5, 5), (1, 4), (4, 12), (2, 3), (1, 4), (4,3), (3,7),(3,4), (2,1), (1,2)]},
            'conv2_kernel_size': {'values': [(3, 3), (3, 7), (5, 5), (2, 2), (1, 1), (3, 2), (2, 3), (4, 2), (2, 4),(4,3),(3,4),(2,1), (1,2)]},
            'pool_size_1': {"values": [(1,1), (2,1), (3,1), (4,1), (1,2), (1,3), (1,4), (2,2), (2,3), (3,3), (3,4), (4,4)]},
            'pool_size_2': {"values": [(1,1), (2,1), (3,1), (4,1), (1,2), (1,3), (1,4), (2,2), (2,3), (3,3), (3,4), (4,4)]},
            'conv2_dimensions': {'values': [10, 20, 30, 50, 80, 100, 200, 400, 800]},
            'conv_out_dimensions': {'values': [50, 100, 150, 200, 400, 600]},
            'linear_out': {'values': [8, 12, 16, 32, 64, 128, 150]},
            'n_mels': {'values': [80]},
            'hop_length': {'values': [44]},
            "dropout": {"values": [0, 0.1, 0.01, 0.5, 0.3, 0.4, 0.2, 0.15]},
            "weight_decay":  {"values": [1e-3, 1e-4, 0, 1e-5, 1e-2]}
        }
    }

    wandb.agent("khvoe3y5", function=sweep_train, count=1000, project="onset-detection-cnn-end-may")
