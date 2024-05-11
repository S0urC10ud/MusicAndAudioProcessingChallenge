from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import librosa
from scipy.io import wavfile
import random

class OnsetDetectionDatasetOld(Dataset):
    def __init__(self, files, all_wavs, all_onsets, sample_length=44100, n_fft=[2048, 1024, 512], hop_length=32, n_mels=200):
        self.files = files
        self.sample_length = sample_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.all_wavs = all_wavs
        self.all_onsets = all_onsets

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        onset_file = audio_file.replace('.wav', '.onsets.gt')
        #sample_rate, audio = wavfile.read(audio_file)
        sample_rate, audio = self.all_wavs[audio_file]
        if audio.dtype.kind == 'i':
            audio = audio.astype(float) / np.iinfo(audio.dtype).max
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if len(audio) > self.sample_length:
            max_offset = len(audio) - self.sample_length
            offset = np.random.randint(0, max_offset)
            audio = audio[offset:offset + self.sample_length]

        spectrograms = []
        for n_fft in self.n_fft:
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            spectrograms.append(log_spectrogram.reshape(1, log_spectrogram.shape[0], log_spectrogram.shape[1]))

        stacked_spectrograms = np.concatenate(spectrograms, axis=0)

        #with open(onset_file, 'r') as f:
        #    onsets = np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)
        onsets = self.all_onsets[audio_file]
        frame_onsets = (onsets - offset / sample_rate) * sample_rate / np.mean(self.hop_length)  # Average hop length for onset calculation
        frame_onsets = frame_onsets[(frame_onsets >= 0) & (frame_onsets < self.sample_length / np.mean(self.hop_length))]

        onset_tensor_size = int(self.sample_length / np.mean(self.hop_length))
        onset_tensor = torch.zeros(onset_tensor_size, dtype=torch.float32)
        onset_tensor[np.minimum(np.floor(frame_onsets).astype(int), onset_tensor_size-1)] = 1.0

        return torch.tensor(stacked_spectrograms, dtype=torch.float32), onset_tensor

class OnsetDetectionDataset(Dataset):
    def __init__(self, files, all_wavs, all_onsets, sample_delta=7, hop_length=256, n_mels=200):
        self.files = files
        self.sample_delta = sample_delta
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.all_wavs = all_wavs
        self.all_onsets = all_onsets

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        sample_rate, stacked_spectrograms = self.all_wavs[audio_file]
        onsets = self.all_onsets[audio_file]
        all_onset_frames = np.floor(onsets/(self.hop_length / sample_rate)).astype(int)

        if random.randint(1,2) == 1:
            #choose randomly
            offset = random.randint(self.sample_delta,stacked_spectrograms.shape[2]- self.sample_delta - 1)

        else:
            # choose some onset offset
            offset=float("inf")
            while offset + self.sample_delta > stacked_spectrograms.shape[2] or offset - self.sample_delta < 0:
                offset = random.choice(all_onset_frames)

        stacked_spectrograms_random_subset = stacked_spectrograms[:,:, offset-self.sample_delta:offset+self.sample_delta + 1]
        
        #is_onset = 1 if len(set(range(offset,offset+3)).intersection(set(all_onset_frames)))>0 else 0
        is_onset = 1 if offset in all_onset_frames or (offset+1) in all_onset_frames or (offset-1) in all_onset_frames else 0
        stacked_spectrograms_random_subset =  torch.tensor(stacked_spectrograms_random_subset, dtype=torch.float32)
        
        assert list(stacked_spectrograms_random_subset.shape) == [3, 80, 15]
        return stacked_spectrograms_random_subset, torch.tensor(is_onset, dtype=torch.float32)
