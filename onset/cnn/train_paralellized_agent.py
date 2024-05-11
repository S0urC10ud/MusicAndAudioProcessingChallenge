import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np
from scipy.io import wavfile
import librosa
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import CustomCNN
import dill
import wandb
from dataset import OnsetDetectionDataset
from functools import partial


def train_test_split(dataset_dir, test_size=0.2, seed=42):
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.wav')]
    np.random.seed(seed)
    np.random.shuffle(files)
    split_idx = int(len(files) * (1 - test_size))
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    return train_files, test_files

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_conv_layers=2, num_fc_layers=2, base_channels=16):
        super(SimpleCNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.base_channels = base_channels

        layers = []
        in_channels = 3
        for i in range(num_conv_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        fc_layers = []
        for i in range(num_fc_layers - 1):
            fc_layers.append(nn.LazyLinear(100))  # Arbitrary number of units, LazyLinear will infer the correct input size
            fc_layers.append(nn.ReLU())
        
        fc_layers.append(nn.LazyLinear(1))  # Number of outputs matches your dataset specifics
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

def train_model(train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data, epochs=300):
    run = wandb.init(project="audio_onset_detection", reinit=True)
    config = run.config

    train_dataset = OnsetDetectionDataset(train_files, wavfile_train_data, onset_train_data, sample_length=2**11, n_mels=config.n_mels, hop_length=config.hop_length)
    test_dataset = OnsetDetectionDataset(test_files, wavfile_test_data, onset_test_data, sample_length=2**11, n_mels=config.n_mels, hop_length=config.hop_length)

    # Use a higher number of workers and enable pin_memory for faster host to GPU transfers
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)


    model = CustomCNN(conv1_kernel_size=config.conv1_kernel_size, conv2_kernel_size=config.conv2_kernel_size, conv2_dimensions = config.conv2_dimensions, conv_out_dimensions = config.conv_out_dimensions, linear_out=config.linear_out)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    model.cuda()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = torch.max(targets, dim=1)[0].reshape(-1,1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in test_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, dim=1)[0].reshape(-1,1))
                total_loss += loss.item()
            avg_test_loss = total_loss / len(test_loader)

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_test_loss})
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}')
        with open("cnn_output.dmp", "wb") as dmp_file:
            dill.dump(model, dmp_file)


from torch.utils.data import DataLoader

sweep_config = {
    'method': 'bayes',  # can be grid, random, or bayes
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'hop_length': {
            'values': [16, 32, 64, 128, 256]
        },
        'conv1_kernel_size': {
            'values': [(12, 3), (8, 2), (4, 2)]
        },
        'conv2_kernel_size': {
            'values': [(3, 3), (5, 5), (7, 7)]
        },
        'conv2_dimensions': {
            'values': [5,10,20,30, 40]
        },
        'conv_out_dimensions': {
            'values': [5,10,20,30, 40, 80]
        },
        'linear_out': {
            'values': [5, 10, 40, 80, 200, 128, 400]
        },
        'n_mels': {
            'values': [50, 100, 128, 300]
        },
        'learning_rate': {
            'values': [0.001, 0.0005, 0.0001, 0.00001]
        },
        'batch_size':{
            'values': [20, 40, 80, 150, 200]
        }
    }
}

if __name__ == '__main__':
    train_files, test_files = train_test_split('data/train_extra_onsets/', test_size=0.2, seed=42)
    wavfile_train_data = {}
    onset_train_data = {}
    for t in tqdm(train_files):
        wavfile_train_data[t] = wavfile.read(t)
        with open(t.replace(".wav", ".onsets.gt"), 'r') as f:
            onset_train_data[t] =  np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)

    wavfile_test_data = {}
    onset_test_data = {}
    for t in tqdm(test_files):
        wavfile_test_data[t] = wavfile.read(t)
        with open(t.replace(".wav", ".onsets.gt"), 'r') as f:
            onset_test_data[t] =  np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)

    model_training_function = partial(train_model, train_files, wavfile_train_data, onset_train_data, test_files, wavfile_test_data, onset_test_data)

    # Using existing sweep ID
    sweep_id = 'kub5z3t0'  # Set your existing sweep ID here
    wandb.agent(sweep_id, model_training_function, project="audio_onset_detection")
