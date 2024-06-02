from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import librosa
from scipy.io import wavfile
import random
import torch.nn.functional as F

class OnsetDetectionDataset(Dataset):
    def __init__(self, files, all_wavs, all_onsets, sample_delta=7, hop_length=256, n_mels=200, onset_one_in=2):
        self.files = files
        self.sample_delta = sample_delta
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.all_wavs = all_wavs
        self.all_onsets = all_onsets
        self.onset_one_in = onset_one_in

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        sample_rate, stacked_spectrograms = self.all_wavs[audio_file]
        onsets = self.all_onsets[audio_file]
        all_onset_frames = np.floor(onsets/(self.hop_length / sample_rate)).astype(int)
        was_between = False
        if random.randint(1, 3) == 1:
            # between two consecutive onsets
            offset = -15
            while offset + self.sample_delta > stacked_spectrograms.shape[2] or offset - self.sample_delta < 0:
                first_id = random.randint(0, len(all_onset_frames)-2)
                offset = int((all_onset_frames[first_id] + all_onset_frames[first_id + 1]) / 2) + random.randint(-8, 8)
            was_between = True
        elif random.randint(1,self.onset_one_in) != 1:
            if random.randint(0,2) == 0:
                #also model close misses as 0
                offset=float("inf")
                while offset + self.sample_delta > stacked_spectrograms.shape[2] or offset - self.sample_delta < 0:
                    offset = random.choice(all_onset_frames)

                offset += random.randint(3,10)*random.choice([-1,1])
                if offset - self.sample_delta < 0:
                    offset += self.sample_delta*2
                elif offset + self.sample_delta + 1 > stacked_spectrograms.shape[2]:
                    offset -= self.sample_delta*2
            else:
                offset = random.randint(self.sample_delta,stacked_spectrograms.shape[2] - self.sample_delta - 1)
                while any(abs(all_onset_frames - offset) < 3):
                    offset = random.randint(self.sample_delta,stacked_spectrograms.shape[2] - self.sample_delta - 1)
        else:
            # choose some onset offset
            offset=float("inf")
            while offset + self.sample_delta > stacked_spectrograms.shape[2] or offset - self.sample_delta < 0:
                offset = random.choice(all_onset_frames)

        stacked_spectrograms_random_subset = stacked_spectrograms[:,:, offset-self.sample_delta:offset+self.sample_delta + 1]
        
        #is_onset = 1 if len(set(range(offset,offset+3)).intersection(set(all_onset_frames)))>0 else 0
        is_onset = 1 if any([(offset + i) in all_onset_frames for i in range(-2, 2)]) else 0
        stacked_spectrograms_random_subset =  torch.tensor(stacked_spectrograms_random_subset, dtype=torch.float32)

        if list(stacked_spectrograms_random_subset.shape) != [3, 80, 15]:
            if offset - self.sample_delta < 0:
                stacked_spectrograms_random_subset = F.pad(stacked_spectrograms_random_subset, (offset - self.sample_delta, 0), "constant", 0)
            elif offset + self.sample_delta >= stacked_spectrograms.shape[2]:
                stacked_spectrograms_random_subset = F.pad(stacked_spectrograms_random_subset, (0, stacked_spectrograms.shape[2] - offset - self.sample_delta + 1), "constant", 0)
        assert list(stacked_spectrograms_random_subset.shape) == [3, 80, 15]
        return stacked_spectrograms_random_subset, torch.tensor(is_onset, dtype=torch.float32)

class OnsetDetectionDatasetOld(Dataset):
    def __init__(self, files, all_wavs, all_onsets, sample_delta=7, hop_length=256, n_mels=200, onset_one_in=2):
        self.files = files
        self.sample_delta = sample_delta
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.all_wavs = all_wavs
        self.all_onsets = all_onsets
        self.onset_one_in = onset_one_in

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        sample_rate, stacked_spectrograms = self.all_wavs[audio_file]
        onsets = self.all_onsets[audio_file]
        all_onset_frames = np.floor(onsets/(self.hop_length / sample_rate)).astype(int)
        
        offset = random.randint(self.sample_delta, stacked_spectrograms.shape[2]-self.sample_delta - 1)

        stacked_spectrograms_random_subset = stacked_spectrograms[:,:, offset-self.sample_delta:offset+self.sample_delta + 1]
        
        #is_onset = 1 if len(set(range(offset,offset+3)).intersection(set(all_onset_frames)))>0 else 0
        is_onset = 1 if any([(offset + i) in all_onset_frames for i in range(-3, 3)]) else 0
        stacked_spectrograms_random_subset =  torch.tensor(stacked_spectrograms_random_subset, dtype=torch.float32)

        if list(stacked_spectrograms_random_subset.shape) != [3, 80, 15]:
            print("had to re-try")
            return self.__getitem__(idx)
        return stacked_spectrograms_random_subset, torch.tensor(is_onset, dtype=torch.float32)
