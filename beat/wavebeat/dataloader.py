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

#torchaudio.set_audio_backend("sox_io")

class BeatDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dir, 
                 audio_sample_rate=22050,
                 target_factor=256,
                 length=88200, 
                 preload=True,
                 subset="train",
                 augment=True,
                 pad_mode='constant',
                 examples_per_epoch=1000):

        self.dir = dir
        self.audio_sample_rate = audio_sample_rate
        self.target_factor = target_factor
        self.target_sample_rate = audio_sample_rate / target_factor
        self.length = length
        self.preload = preload
        self.subset = subset
        self.augment = augment
        if self.augment:
            print("Also augmenting!")
        self.pad_mode = pad_mode
        self.examples_per_epoch = examples_per_epoch

        self.target_length = int(self.length / self.target_factor)


        self.audio_files = glob.glob(os.path.join(self.dir, "*.wav"))

        random.seed(43)
        random.shuffle(self.audio_files) # shuffle them

        if self.subset == "train":
            start = 0
            stop = int(len(self.audio_files) * 0.8)
        elif self.subset == "val":
            start = int(len(self.audio_files) * 0.8)
            stop = int(len(self.audio_files) * 0.9)
        elif self.subset == "test":
            start = int(len(self.audio_files) * 0.9)
            stop = None
        elif self.subset in ["full-train", "full-val"]:
            start = 0
            stop = None


        self.audio_files = self.audio_files[start:stop]
        print(f"Selected {len(self.audio_files)} files for {self.subset}")

        self.annot_files = []
        for audio_file in self.audio_files:
            #corresponding annotation         
            filename = os.path.basename(audio_file).replace(".wav", ".beats.gt")
            self.annot_files.append(os.path.join(self.dir, filename))

        self.data = [] # when preloading store audio data and metadata
        if self.preload:
            for audio_filename, annot_filename in tqdm(zip(self.audio_files, self.annot_files), 
                                                        total=len(self.audio_files), 
                                                        ncols=80):
                    audio, target, metadata = self.load_data(audio_filename, annot_filename)
                    self.data.append((audio, target, metadata))

    def __len__(self):
        if self.subset in ["test", "val", "full-val", "full-test"]:
            length = len(self.audio_files)
        else:
            length = self.examples_per_epoch
        return length

    def __getitem__(self, idx):
        if self.preload:
            audio, target, metadata = self.data[idx % len(self.audio_files)]
        else:
            # get metadata of example
            audio_filename = self.audio_files[idx % len(self.audio_files)]
            annot_filename = self.annot_files[idx % len(self.audio_files)]
            audio, target, metadata = self.load_data(audio_filename, annot_filename)

        # do all processing in float32 not float16
        audio = audio.float()
        target = target.float()

        # apply augmentations 
        if self.augment: 
            audio, target = self.apply_augmentations(audio, target) # TODO: Rework this Martin

        N_audio = audio.shape[-1]   # audio samples
        N_target = target.shape[-1] # target samples

        # random crop of the audio and target if larger than desired
        if (N_audio > self.length or N_target > self.target_length) and self.subset not in ['val', 'test', 'full-val']:
            audio_start = np.random.randint(0, N_audio - self.length - 1)
            audio_stop  = audio_start + self.length
            target_start = int(audio_start / self.target_factor)
            target_stop = int(audio_stop / self.target_factor)
            audio = audio[:,audio_start:audio_stop]
            target = target[:,target_start:target_stop]

        # pad the audio and target is shorter than desired
        if audio.shape[-1] < self.length and self.subset not in ['val', 'test', 'full-val']: 
            pad_size = self.length - audio.shape[-1]
            padl = pad_size // 2
            padr = pad_size - padl
            audio = torch.nn.functional.pad(audio, (padl, padr), mode=self.pad_mode)

        if target.shape[-1] < self.target_length and self.subset not in ['val', 'test', 'full-val']: 
            pad_size = self.target_length - target.shape[-1]
            padl = pad_size // 2
            padr = pad_size - padl
            target = torch.nn.functional.pad(target, (padl, padr), mode=self.pad_mode)

        if self.subset in ["train", "full-train"]:
            return audio, target
        elif self.subset in ["val", "test", "full-val"]:
            # this will only work with batch size = 1
            return audio, target, metadata
        else:
            raise RuntimeError(f"Invalid subset: `{self.subset}`")

    def load_data(self, audio_filename, annot_filename):
        audio, sr = torchaudio.load(audio_filename)
        audio = audio.float()
        # resample if needed
        if sr != self.audio_sample_rate:
            audio = julius.resample_frac(audio, sr, self.audio_sample_rate)   

        # normalize all audio inputs -1 to 1
        audio /= audio.abs().max()

        # now get the annotation information
        beat_samples, downbeat_samples, beat_indices, time_signature = self.load_annot(annot_filename)

        # convert beat_samples to beat_seconds
        beat_sec = np.array(beat_samples) / self.audio_sample_rate
        downbeat_sec = np.array(downbeat_samples) / self.audio_sample_rate

        t = audio.shape[-1]/self.audio_sample_rate # audio length in sec
        N = int(t * self.target_sample_rate) + 1   # target length in samples
        target = torch.zeros(2,N)

        # now convert from seconds to new sample rate
        beat_samples = np.array(beat_sec * self.target_sample_rate)
        downbeat_samples = np.array(downbeat_sec * self.target_sample_rate)

        # strip beats beyond file end
        beat_samples = beat_samples[beat_samples < N]
        downbeat_samples = downbeat_samples[downbeat_samples < N]

        beat_samples = beat_samples.astype(int)
        downbeat_samples = downbeat_samples.astype(int)

        target[0,beat_samples] = 1  # first channel is beats
        target[1,downbeat_samples] = 1  # second channel is downbeats

        metadata = {
            "Filename" : audio_filename,
            "Time signature" : time_signature
        }

        return audio, target, metadata

    def load_annot(self, filename):

        with open(filename, 'r') as fp:
            lines = fp.readlines()
        
        beat_samples = [] # array of samples containing beats
        downbeat_samples = [] # array of samples containing downbeats (1)
        beat_indices = [] # array of beat type one-hot encoded  
        time_signature = None # estimated time signature (only 3/4 or 4/4)

        for line in lines:
            line = line.strip('\n')
            time_sec, beat_info = line.split('\t')
            takt, beat = beat_info.split(".")
            
            beat = int(beat)

            # convert seconds to samples
            beat_time_samples = int(float(time_sec) * (self.audio_sample_rate))

            beat_samples.append(beat_time_samples)
            beat_indices.append(beat)

            if beat == 1:
                downbeat_time_samples = int(float(time_sec) * (self.audio_sample_rate))
                downbeat_samples.append(downbeat_time_samples)

        # guess at the time signature
        if np.max(beat_indices) == 2:
            time_signature = "2/4"
        elif np.max(beat_indices) == 3:
            time_signature = "3/4"
        elif np.max(beat_indices) == 4:
            time_signature = "4/4"
        else:
            time_signature = "?"

        return beat_samples, downbeat_samples, beat_indices, time_signature

    def apply_augmentations(self, audio, target):
        # random gain from 0dB to -6 dB
        #if np.random.rand() < 0.2:      
        #    #sgn = np.random.choice([-1,1])
        #    audio = audio * (10**((-1 * np.random.rand() * 6)/20))   

        # phase inversion
        if np.random.rand() < 0.5:      
            audio = -audio                              

        # shift targets forward/back max 70ms
        if np.random.rand() < 0.3:      
            
            # in this method we shift each beat and downbeat by a random amount
            max_shift = int(0.045 * self.target_sample_rate)

            beat_ind = torch.logical_and(target[0,:] == 1, target[1,:] != 1).nonzero(as_tuple=False) # all beats EXCEPT downbeats
            dbeat_ind = (target[1,:] == 1).nonzero(as_tuple=False)

            # shift just the downbeats
            dbeat_shifts = torch.normal(0.0, max_shift/2, size=(1,dbeat_ind.shape[-1]))
            dbeat_ind += dbeat_shifts.long()

            # now shift the non-downbeats 
            beat_shifts = torch.normal(0.0, max_shift/2, size=(1,beat_ind.shape[-1]))
            beat_ind += beat_shifts.long()

            # ensure we have no beats beyond max index
            beat_ind = beat_ind[beat_ind < target.shape[-1]]
            dbeat_ind = dbeat_ind[dbeat_ind < target.shape[-1]]  

            # now convert indices back to target vector
            shifted_target = torch.zeros(2,target.shape[-1])
            shifted_target[0,beat_ind] = 1
            shifted_target[0,dbeat_ind] = 1 # set also downbeats on first channel
            shifted_target[1,dbeat_ind] = 1

            target = shifted_target

        # apply a lowpass filter
        if np.random.rand() < 0.1:
            cutoff = (np.random.rand() * 4000) + 4000
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="lowpass", 
                                      fs=self.audio_sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # apply a highpass filter
        if np.random.rand() < 0.1:
            cutoff = (np.random.rand() * 1000) + 20
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="highpass", 
                                      fs=self.audio_sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
            audio = torch.from_numpy(audio_filtered.astype('float32'))
        
        # add white noise
        if np.random.rand() < 0.05:
            wn = (torch.rand(audio.shape) * 2) - 1
            g = 10**(-(np.random.rand() * 20) - 12)/20
            audio = audio + (g * wn)

        # apply nonlinear distortion 
        if np.random.rand() < 0.2:   
            g = 10**((np.random.rand() * 12)/20)   
            audio = torch.tanh(audio)    
        
        # normalize the audio
        audio /= audio.float().abs().max()

        return audio, target
