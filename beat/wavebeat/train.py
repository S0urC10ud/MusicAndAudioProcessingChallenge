import os
import glob
import torch
import torchsummary
from itertools import product
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from model import dsTCNModel
from dataloader import BeatDataset

torch.backends.cudnn.benchmark = True

pl.seed_everything(12)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min' 
)

trainer = pl.Trainer(callbacks=[checkpoint_callback])

train_dataset = BeatDataset("data/train/",
                            sample_rate=22050,
                            downsample_factor=256,
                            data_subset="train",
                            preload_data=True,
                            segment_length=2097152 )

val_dataset = BeatDataset("data/train/",
                            sample_rate=22050,
                            downsample_factor=256,
                            data_subset="val",
                            preload_data=True,
                            segment_length=2097152 )


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

model = dsTCNModel()
torchsummary.summary(model, [(1,65536)], device="cpu")
trainer.fit(model, train_dataloader, val_dataloader)