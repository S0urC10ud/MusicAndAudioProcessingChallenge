import os
import glob
import torch
import torchaudio
import pytorch_lightning as pl
from argparse import ArgumentParser

from model import dsTCNModel

parser = ArgumentParser()
model="checkpoints/"
input_file = "data/train/ff123_beo1test.wav"

# find the checkpoint path
ckpts = glob.glob(os.path.join(model, "*.ckpt"))
if len(ckpts) < 1:
    raise RuntimeError(f"No checkpoints found in {model}.")
else:
    ckpt_path = ckpts[-1]

# construct the model, and load weights from checkpoint
print(f"Loading from checkpoint {ckpt_path}")
model = dsTCNModel.load_from_checkpoint(ckpt_path)

# set model to eval mode
model.eval()

# get the locations of the beats and downbeats
beats, downbeats = model.predict_beats(input_file, use_gpu=True)

# print some results to terminal
print(f"Beats found in {input_file}")
print("-" * 32)
for beat in beats:
    print(f"{beat:0.2f}")

print()
print(f"Downbeats found in {input_file}")
print("-" * 32)
for downbeat in downbeats:
    print(f"{downbeat:0.2f}")