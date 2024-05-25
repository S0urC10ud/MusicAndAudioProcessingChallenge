import os
import glob
import torch
import torchsummary
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint
from model import dsTCNModel
from dataloader import BeatDataset

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Monitoring validation loss for checkpointing
    dirpath='checkpoints/',  # Directory where checkpoints will be saved
    filename='model-{epoch:02d}-{val_loss:.2f}',  # Naming format for checkpoints
    save_top_k=3,  # Number of top models to keep
    mode='min'  # Mode for comparison, here it's minimizing the monitored quantity (validation loss)
)

# set the seed
pl.seed_everything(42)
# create the trainer
trainer = pl.Trainer(callbacks=[checkpoint_callback])

train_dataset = BeatDataset("data/train/",
                                audio_sample_rate=22050,
                                target_factor=256,
                                subset="train",
                                augment=True,
                                preload=True,
                                length=65536*12)

val_dataset = BeatDataset("data/train/",
                                audio_sample_rate=22050,
                                target_factor=256,
                                subset="val",
                                augment=False,
                                preload=True,
                                length=65536*12)


train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True,
                                                batch_size=16,
                                                num_workers=4,
                                                pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                            shuffle=False,
                                            batch_size=1,
                                            num_workers=1,
                                            pin_memory=False)    

# create the model with args
model = dsTCNModel()

# summary 
torchsummary.summary(model, [(1,65536)], device="cpu")

# train!
trainer.fit(model, train_dataloader, val_dataloader)