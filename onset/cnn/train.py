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


def train_test_split(dataset_dir, test_size=0.2, seed=42):
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.wav')]
    np.random.seed(seed)
    np.random.shuffle(files)
    split_idx = int(len(files) * (1 - test_size))
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    return train_files, test_files


def train_model(train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data, config, epochs=25000):
    train_dataset = OnsetDetectionDataset(train_files, wavfile_train_data, onset_train_data, sample_delta=7, n_mels=config.n_mels, hop_length=config.hop_length)
    test_dataset = OnsetDetectionDataset(test_files, wavfile_test_data, onset_test_data, sample_delta=7, n_mels=config.n_mels, hop_length=config.hop_length)

    # Use a higher number of workers and enable pin_memory for faster host to GPU transfers
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)


    model = CustomCNN(conv1_kernel_size=config.conv1_kernel_size, conv2_kernel_size=config.conv2_kernel_size, conv2_dimensions = config.conv2_dimensions, conv_out_dimensions = config.conv_out_dimensions, linear_out=config.linear_out)
    model.cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        all_train_preds, all_train_targets = [], []

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.reshape(-1,1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # For accuracy and confusion matrix
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
                loss = criterion(outputs, targets.reshape(-1,1))
                total_test_loss += loss.item()

                # For accuracy and confusion matrix
                test_preds = (outputs > 0.5).to(dtype=int).squeeze()
                all_test_preds.extend(test_preds.cpu().numpy())
                all_test_targets.extend(targets.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = accuracy_score(all_test_targets, all_test_preds)
        test_confusion_mat = confusion_matrix(all_test_targets, all_test_preds)


        # Save the model periodically or at certain epochs
        if (epoch+1) % 25 == 0:  # Every 25 epochs
            print("-"*40)
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            print("Train Confusion Matrix:\n", train_confusion_mat)
            print("Test Confusion Matrix:\n", test_confusion_mat)
            print("-"*40)
            with open(f"cnn_output.dmp", "wb") as dmp_file:
                dill.dump(model, dmp_file)

current_config = DotAccessibleDict({
    'hop_length': 44,
    'conv1_kernel_size': (3,7),
    'conv2_kernel_size': (3,3),
    'conv2_dimensions': 20,
    'conv_out_dimensions': 50,
    'linear_out': 256,
    'n_mels': 80,
    'learning_rate': 0.00001,
    'batch_size': 40
})

reload_data = False

if __name__ == '__main__':
    train_files, test_files = train_test_split('data/train_extra_onsets/', test_size=0.2, seed=42)
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
                spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=current_config.hop_length, n_mels=current_config.n_mels)
                log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
                full_spectrograms.append(log_spectrogram)

            wavfile_train_data[t] = sample_rate, np.stack(full_spectrograms, axis=0) 

            with open(t.replace(".wav", ".onsets.gt"), 'r') as f:
                onset_train_data[t] =  np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
            
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
                log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
                full_spectrograms.append(log_spectrogram)

            wavfile_test_data[t] = sample_rate, np.stack(full_spectrograms, axis=0) 

            with open(t.replace(".wav", ".onsets.gt"), 'r') as f:
                onset_test_data[t] =  np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
        with open("onset_dataset.dill", "wb") as onset_dataset_file:
            dill.dump((train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data), onset_dataset_file)
    else:
        with open("onset_dataset.dill", "rb") as onset_dataset_file:
            train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data = dill.load(onset_dataset_file)
        
    train_model(train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data, current_config)