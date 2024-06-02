from beat.wavebeat.loss import BCFELoss
from beat.wavebeat.eval import evaluate, find_beats
from beat.wavebeat.plot import plot_activations, make_table, plot_histogram
import torch
from argparse import ArgumentParser
import os
import torch
import julius
import torchaudio
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser


def center_crop(x, length: int):
    start = (x.shape[-1]-length)//2
    stop  = start + length
    return x[...,start:stop]

def causal_crop(x, length: int):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[...,start:stop]

class Base(pl.LightningModule):
    """ Base module with train and validation loops from the original repo

        Args:
            nparams (int): Number of conditioning parameters.
            lr (float, optional): Learning rate. Default: 3e-4
            train_loss (str, optional): Training loss function from ['l1', 'stft', 'l1+stft']. Default: 'l1+stft'
            save_dir (str): Path to save audio examples from validation to disk. Default: None
            num_examples (int, optional): Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self, 
                    lr = 5e-5, 
                    save_dir = None,
                    num_examples = 4,
                    **kwargs):
        super(Base, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.save_dir = save_dir
        self.num_examples = num_examples
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()
        self.bcfe = BCFELoss()
        self.validation_outputs = []
        self.target_sample_rate = 22050/256.0
        self.save_dir= "lightning_talks/"
        self.train_loss=float("inf")


    def forward(self, x):
        pass

    @torch.jit.unused  
    def predict_beats(self, filename):
        audio, sr = torchaudio.load(filename)

        # resample to 22.05 kHz if needed
        if sr != 22050:
            audio = julius.resample_frac(audio, sr, 22050)   

        if audio.shape[0] > 1:
            print("Loaded multichannel audio. Summing to mono...")
            audio = audio.mean(dim=0, keepdim=True)

        # normalize the audio
        audio /= audio.abs().max()
        audio = audio.unsqueeze(0)

        audio = audio.to("cuda:0")
        self.to("cuda:0")

        with torch.no_grad():
            pred = torch.sigmoid(self(audio))

        p_beats = pred[0,0,:]
        p_downbeats = pred[0,1,:]

        _, beats, _ = find_beats(p_beats.detach().cpu().numpy(), 
                                    p_beats.detach().cpu().numpy(), 
                                    beat_type="beat",
                                    sample_rate=self.target_sample_rate)

        _, downbeats, _ = find_beats(p_downbeats.detach().cpu().numpy(), 
                                        p_downbeats.detach().cpu().numpy(), 
                                        beat_type="downbeat",
                                        sample_rate=self.target_sample_rate)

        return beats, downbeats

    @torch.jit.unused   
    def training_step(self, batch, batch_idx):
        input, target = batch

        # pass the input thrgouh the mode
        pred = self(input)

        # crop the input and target signals
        target = center_crop(target, pred.shape[-1])

        # compute the error using appropriate loss      
        #loss, _, _ = self.gbce(pred, target)
        loss, _, _ = self.bcfe(pred, target)
        self.train_loss = loss
        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target, metadata = batch
        pred = self(input)
        target_crop = center_crop(target, pred.shape[-1])
        loss, _, _ = self.bcfe(pred, target_crop)
        self.log('val_loss', loss)
        pred = torch.sigmoid(pred)

        outputs = {
            "input" : input.cpu(),
            "target": target_crop.cpu(),
            "pred"  : pred.cpu(),
            "Filename" : metadata['Filename'],
            "Time signature" : metadata['Time signature']
            }
        self.validation_outputs.append(outputs)
        return outputs

    @torch.jit.unused
    def on_validation_epoch_start(self):

        self.validation_outputs = []
    @torch.jit.unused
    def on_validation_epoch_end(self):
        # flatten the output validation step dicts to a single dict
        outputs = {
            "input" : [],
            "target" : [],
            "pred" : [],
            "Filename" : [],
            "Time signature" : []}

        metadata_keys = ["Filename", "Time signature"]

        for out in self.validation_outputs:
            for key, val in out.items():
                if key not in metadata_keys:
                    bs = val.shape[0]
                else:
                    bs = len(val)
                for bidx in np.arange(bs):
                    if key not in metadata_keys:
                        outputs[key].append(val[bidx,...])
                    else:
                        outputs[key].append(val[bidx])

        example_indices = np.arange(len(outputs["input"]))
        rand_indices = np.random.choice(example_indices,
                                        replace=False,
                                        size=np.min([len(outputs["input"]), 4]))

        # compute metrics 
        songs = []
        beat_f1_scores = []
        downbeat_f1_scores = []

        for idx in np.arange(len(outputs["input"])):
            t = outputs["target"][idx].squeeze()
            p = outputs["pred"][idx].squeeze()
            f = outputs["Filename"][idx]
            s = outputs["Time signature"][idx]

            beat_scores, downbeat_scores = evaluate(p, t, self.target_sample_rate)

            songs.append({
                "Filename" : f,
                "Time signature" : s,
                "Beat F-measure" : beat_scores['F-measure'],
                "Downbeat F-measure" : downbeat_scores['F-measure'],

            })

            beat_f1_scores.append(beat_scores['F-measure'])
            downbeat_f1_scores.append(downbeat_scores['F-measure'])

        beat_f_measure = np.mean(beat_f1_scores)
        downbeat_f_measure = np.mean(downbeat_f1_scores)
        self.log('val_loss/Beat F-measure', torch.tensor(beat_f_measure))
        self.log('val_loss/Downbeat F-measure', torch.tensor(downbeat_f_measure))
        self.log('val_loss/Joint F-measure', torch.tensor(np.mean([beat_f_measure,downbeat_f_measure])))


        self.logger.experiment.add_text("perf", 
                                        make_table(songs),
                                        self.global_step)
    
        self.logger.experiment.add_image(f"hist/F-measure",
                                         plot_histogram(songs),
                                         self.global_step)
 
        for idx, rand_idx in enumerate(list(rand_indices)):
            i = outputs["input"][rand_idx].squeeze()
            t = outputs["target"][rand_idx].squeeze()
            p = outputs["pred"][rand_idx].squeeze()
            f = outputs["Filename"][idx]
            s = outputs["Time signature"][idx]

            t_beats = t[0,:]
            t_downbeats = t[1,:]
            p_beats = p[0,:]
            p_downbeats = p[1,:]

            ref_beats, est_beats, est_sm = find_beats(t_beats.numpy(), 
                                                      p_beats.numpy(), 
                                                      beat_type="beat",
                                                      sample_rate=self.target_sample_rate)

            ref_downbeats, est_downbeats, est_downbeat_sm = find_beats(t_downbeats.numpy(), 
                                                                       p_downbeats.numpy(), 
                                                                       beat_type="downbeat",
                                                                       sample_rate=self.target_sample_rate)
            # log audio examples
            self.logger.experiment.add_audio(f"input/{idx}",  
                                             i, self.global_step, 
                                             sample_rate=22050)

            # log beats plots
            self.logger.experiment.add_image(f"act/{idx}",
                                             plot_activations(ref_beats, 
                                                              est_beats, 
                                                              est_sm,
                                                              self.target_sample_rate,
                                                              ref_downbeats=ref_downbeats,
                                                              est_downbeats=est_downbeats,
                                                              est_downbeats_sm=est_downbeat_sm,
                                                              song_name=f),
                                             self.global_step)

            if self.save_dir is not None:
                if not os.path.isdir(self.save_dir):
                    os.makedirs(self.save_dir)

                input_filename = os.path.join(self.save_dir, f"{idx}-input.wav")
                target_filename = os.path.join(self.save_dir, f"{idx}-target.wav")

                if not os.path.isfile(input_filename):
                    torchaudio.save(input_filename, 
                                    i.clone().detach().view(1, -1).float(),
                                    sample_rate=22050)

                if not os.path.isfile(target_filename):
                    torchaudio.save(target_filename,
                                    t.clone().detach().view(1, -1).float(),
                                    sample_rate=22050)

                torchaudio.save(os.path.join(self.save_dir, 
                                f"{idx}-pred-{self.train_loss}.wav"), 
                                p.clone().detach().view(1, -1).float(),
                                sample_rate=22050)

    @torch.jit.unused
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.jit.unused
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)                                                      
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss/Joint F-measure',
            'gradient_clip_val': 4.0
        }
    
    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-4)
        # --- vadliation related ---
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--num_examples', type=int, default=4)

        return parser


class dsTCNBlock(torch.nn.Module):
    def __init__(self, 
                in_ch, 
                out_ch, 
                kernel_size, 
                stride=1,
                dilation=1,
                norm_type=None):
        super(dsTCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type

        pad_value =  ((kernel_size-1) * dilation) // 2

        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     dilation=dilation,
                                     padding=pad_value)
        self.act1 = torch.nn.PReLU(out_ch)

        if norm_type == "BatchNorm":
            self.norm1 = torch.nn.BatchNorm1d(out_ch)
            self.res_norm = torch.nn.BatchNorm1d(out_ch)
        else:
            self.norm1 = None
            self.res_norm = None

        self.res_conv = torch.nn.Conv1d(in_ch, 
                                        out_ch, 
                                        kernel_size=1, 
                                        stride=stride)

    def forward(self, x):
        x = x.cuda()
        x_res = x
        
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.act1(x)

        x_res = self.res_conv(x_res)
        if self.res_norm is not None:
            x_res = self.res_norm(x_res)

        return x + x_res

class dsTCNModel(Base):

    def __init__(self, 
                 ninputs=1,
                 noutputs=2,
                 nblocks=10, 
                 kernel_size=3, 
                 stride=2,
                 dilation_growth=8, 
                 channel_growth=32,
                 channel_width=32,
                 stack_size=4,
                 norm_type='BatchNorm'):
        super(dsTCNModel, self).__init__()
        self.save_hyperparameters()
        self.nblocks = nblocks
        self.kernel_size = kernel_size
        self.stride = stride
        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = ninputs if n == 0 else out_ch 
            out_ch = channel_width if n == 0 else in_ch + channel_growth
            dilation = dilation_growth ** (n % stack_size)

            self.blocks.append(dsTCNBlock(
                in_ch, 
                out_ch,
                kernel_size,
                stride,
                dilation,
                norm_type
            ))

        self.output = torch.nn.Conv1d(out_ch, 
                                      noutputs, 
                                      kernel_size=1)

    def forward(self, x):

        for block in self.blocks:
            x = block(x)
        
        x = self.output(x)

        return x

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = 0
        for n in range(self.nblocks):
            rf += (self.kernel_size - 1) * \
                  (self.nblocks * self.stride)
        return rf

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-2)
        parser.add_argument('--patience', type=int, default=40)
        # --- model related ---
        parser.add_argument('--ninputs', type=int, default=1)
        parser.add_argument('--noutputs', type=int, default=2)
        parser.add_argument('--nblocks', type=int, default=8)
        parser.add_argument('--kernel_size', type=int, default=15)
        parser.add_argument('--stride', type=int, default=2)
        parser.add_argument('--dilation_growth', type=int, default=8)
        parser.add_argument('--channel_growth', type=int, default=1)
        parser.add_argument('--channel_width', type=int, default=32)
        parser.add_argument('--stack_size', type=int, default=4)
        parser.add_argument('--grouped', default=False, action='store_true')
        parser.add_argument('--causal', default=False, action="store_true")
        parser.add_argument('--skip_connections', default=False, action="store_true")
        parser.add_argument('--norm_type', type=str, default='BatchNorm')

        return parser