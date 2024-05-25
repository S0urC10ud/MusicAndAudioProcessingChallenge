import os
import sys
import glob
import torch
import julius
import random
import torchaudio
import numpy as np
import scipy.signal
from tqdm import tqdm

# torchaudio.set_audio_backend("sox_io")

class RhythmDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 directory, 
                 sample_rate=22050,
                 downsample_factor=256,
                 segment_length=88200, 
                 preload_data=True,
                 data_subset="train",
                 padding_mode='constant',
                 samples_per_epoch=1000):

        self.directory = directory
        self.sample_rate = sample_rate
        self.downsample_factor = downsample_factor
        self.downsampled_rate = sample_rate // downsample_factor
        self.segment_length = segment_length
        self.preload_data = preload_data
        self.data_subset = data_subset
        self.padding_mode = padding_mode
        self.samples_per_epoch = samples_per_epoch

        self.target_length = segment_length // downsample_factor

        self.audio_file_paths = glob.glob(os.path.join(self.directory, "*.wav"))

        random.seed(43)
        random.shuffle(self.audio_file_paths)  # shuffle files

        if self.data_subset == "train":
            start_idx = 0
            end_idx = int(len(self.audio_file_paths) * 0.8)
        elif self.data_subset == "val":
            start_idx = int(len(self.audio_file_paths) * 0.8)
            end_idx = int(len(self.audio_file_paths) * 0.9)
        elif self.data_subset == "test":
            start_idx = int(len(self.audio_file_paths) * 0.9)
            end_idx = None
        elif self.data_subset in ["full-train", "full-val"]:
            start_idx = 0
            end_idx = None

        self.audio_file_paths = self.audio_file_paths[start_idx:end_idx]
        print(f"Selected {len(self.audio_file_paths)} files for {self.data_subset}")

        self.annotation_file_paths = []
        for audio_file in self.audio_file_paths:
            filename = os.path.basename(audio_file).replace(".wav", ".beats.gt")
            self.annotation_file_paths.append(os.path.join(self.directory, filename))

        self.loaded_data = []  # for storing preloaded data
        if self.preload_data:
            for audio_fp, annot_fp in tqdm(zip(self.audio_file_paths, self.annotation_file_paths), 
                                           total=len(self.audio_file_paths), 
                                           ncols=80):
                audio, target, metadata = self._load_data(audio_fp, annot_fp)
                self.loaded_data.append((audio, target, metadata))

    def __len__(self):
        if self.data_subset in ["test", "val", "full-val", "full-test"]:
            return len(self.audio_file_paths)
        else:
            return self.samples_per_epoch

    def __getitem__(self, idx):
        if self.preload_data:
            audio, target, metadata = self.loaded_data[idx % len(self.audio_file_paths)]
        else:
            audio_fp = self.audio_file_paths[idx % len(self.audio_file_paths)]
            annot_fp = self.annotation_file_paths[idx % len(self.audio_file_paths)]
            audio, target, metadata = self._load_data(audio_fp, annot_fp)

        audio = audio.float()
        target = target.float()

        audio_samples = audio.shape[-1]
        target_samples = target.shape[-1]

        if (audio_samples > self.segment_length or target_samples > self.target_length) and self.data_subset not in ['val', 'test', 'full-val']:
            audio_start = np.random.randint(0, audio_samples - self.segment_length - 1)
            audio_stop  = audio_start + self.segment_length
            target_start = audio_start // self.downsample_factor
            target_stop = audio_stop // self.downsample_factor
            audio = audio[:, audio_start:audio_stop]
            target = target[:, target_start:target_stop]

        if audio.shape[-1] < self.segment_length and self.data_subset not in ['val', 'test', 'full-val']: 
            pad_size = self.segment_length - audio.shape[-1]
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            audio = torch.nn.functional.pad(audio, (pad_left, pad_right), mode=self.padding_mode)

        if target.shape[-1] < self.target_length and self.data_subset not in ['val', 'test', 'full-val']: 
            pad_size = self.target_length - target.shape[-1]
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            target = torch.nn.functional.pad(target, (pad_left, pad_right), mode=self.padding_mode)

        if self.data_subset in ["train", "full-train"]:
            return audio, target
        elif self.data_subset in ["val", "test", "full-val"]:
            return audio, target, metadata
        else:
            raise RuntimeError(f"Invalid subset: `{self.data_subset}`")

    def _load_data(self, audio_filepath, annot_filepath):
        audio, sr = torchaudio.load(audio_filepath)
        audio = audio.float()
        if sr != self.sample_rate:
            audio = julius.resample_frac(audio, sr, self.sample_rate)   

        audio /= audio.abs().max()

        beat_samples, downbeat_samples, beat_indices, time_signature = self._load_annotation(annot_filepath)

        beat_sec = np.array(beat_samples) / self.sample_rate
        downbeat_sec = np.array(downbeat_samples) / self.sample_rate

        audio_length_sec = audio.shape[-1] / self.sample_rate
        num_target_samples = int(audio_length_sec * self.downsampled_rate) + 1
        target = torch.zeros(2, num_target_samples)

        beat_samples = np.array(beat_sec * self.downsampled_rate)
        downbeat_samples = np.array(downbeat_sec * self.downsampled_rate)

        beat_samples = beat_samples[beat_samples < num_target_samples]
        downbeat_samples = downbeat_samples[downbeat_samples < num_target_samples]

        beat_samples = beat_samples.astype(int)
        downbeat_samples = downbeat_samples.astype(int)

        target[0, beat_samples] = 1
        target[1, downbeat_samples] = 1

        metadata = {
            "Filename": audio_filepath,
            "Time signature": time_signature
        }

        return audio, target, metadata

    def _load_annotation(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        beat_samples = []
        downbeat_samples = []
        beat_indices = []
        time_signature = None

        for line in lines:
            line = line.strip()
            time_sec, beat_info = line.split('\t')
            takt, beat = beat_info.split(".")
            
            beat = int(beat)
            beat_time_samples = int(float(time_sec) * self.sample_rate)

            beat_samples.append(beat_time_samples)
            beat_indices.append(beat)

            if beat == 1:
                downbeat_samples.append(beat_time_samples)

        if np.max(beat_indices) == 2:
            time_signature = "2/4"
        elif np.max(beat_indices) == 3:
            time_signature = "3/4"
        elif np.max(beat_indices) == 4:
            time_signature = "4/4"
        else:
            time_signature = "?"

        return beat_samples, downbeat_samples, beat_indices, time_signature