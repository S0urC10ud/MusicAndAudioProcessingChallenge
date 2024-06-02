import os
import glob
import torch
import torchaudio
import pytorch_lightning as pl
from argparse import ArgumentParser

from model import dsTCNModel

checkpoints_dir="checkpoints/"

def apply_wavebeat(filename):

    # find the checkpoint path
    ckpts = glob.glob(os.path.join(checkpoints_dir, "*.ckpt"))
    if len(ckpts) < 1:
        raise RuntimeError(f"No checkpoints found in {checkpoints_dir}.")
    else:
        ckpt_path = ckpts[-1]

    model = dsTCNModel.load_from_checkpoint(ckpt_path)

    model.eval()
    beats, downbeats = model.predict_beats(filename)

    resulting_beats = beats
    if len(beats)/beats[-1] * 60 > 120:
        print("Using only downbeats")
        resulting_beats = downbeats
    return resulting_beats
