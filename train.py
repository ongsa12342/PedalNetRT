import pytorch_lightning as pl
import argparse
import sys

from logger import DynamicWandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import PedalNet
from prepare import prepare
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import wandb

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import wandb
import pytorch_lightning as pl
import torch

class SpectrogramCallback(pl.Callback):
    def __init__(self, every_n_epochs=100, output_dir='./spectrograms'):
        """
        Args:
            every_n_epochs (int): Frequency (in epochs) at which to create and log the plots.
            output_dir (str): Directory to save the spectrogram images.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        # Iterate through the entire validation dataloader.
         # Check if val_dataloaders is a list or a single DataLoader
        if isinstance(trainer.val_dataloaders, list):
            val_loader = trainer.val_dataloaders[0]
        else:
            val_loader = trainer.val_dataloaders

        for batch_idx, (x, y) in enumerate(val_loader):
            # Ensure data is on the proper device.
            x = x.to(pl_module.device)
            # Get model predictions.
            y_pred = pl_module(x)
            
            # Process each sample in the batch.
            for i in range(x.size(0)):
                # Convert tensors to numpy arrays for plotting.
                y_np = y[i].squeeze().cpu().numpy()
                y_pred_np = y_pred[i].squeeze().detach().cpu().numpy()

                # Define file paths.
                gt_path = os.path.join(self.output_dir, f'epoch_{epoch}_batch_{batch_idx}_sample_{i}_gt.png')
                pred_path = os.path.join(self.output_dir, f'epoch_{epoch}_batch_{batch_idx}_sample_{i}_pred.png')
                
                # Plot and save ground truth and prediction spectrograms.
                self.plot_and_save_spectrogram(y_np, gt_path, title='Ground Truth Spectrogram')
                self.plot_and_save_spectrogram(y_pred_np, pred_path, title='Prediction Spectrogram')

                # # Log images to WandB if using WandBLogger.
                # if trainer.logger is not None and hasattr(trainer.logger, 'experiment'):
                #     trainer.logger.experiment.log({
                #         f'Ground Truth Spectrogram (batch {batch_idx} sample {i})': wandb.Image(gt_path),
                #         f'Prediction Spectrogram (batch {batch_idx} sample {i})': wandb.Image(pred_path),
                #         'epoch': epoch
                #     })

    def plot_and_save_spectrogram(self, audio, save_path, title='Spectrogram'):
        """
        Generates a spectrogram from an audio signal, saves the plot, and then closes the figure.
        """
        # Compute the Short-Time Fourier Transform (STFT)
        D = np.abs(librosa.stft(audio))
        # Convert amplitude to decibels
        D_db = librosa.amplitude_to_db(D, ref=np.max)
        # Create the plot.
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D_db, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        # Save the figure and close it to free up memory.
        plt.savefig(save_path)
        plt.close()


def main(args):
    """
    This trains the PedalNet model to match the output data from the input data.

    When you resume training from an existing model, you can override hparams such as
        max_epochs, batch_size, or learning_rate. Note that changing num_channels,
        dilation_depth, num_repeat, or kernel_size will change the shape of the WaveNet
        model and is not advised.

    """

    prepare(args)
    model = PedalNet(vars(args))

    # Configure accelerator and devices
    if args.cpu:
        accelerator = "cpu"
        devices = 1
    elif args.tpu_cores:
        accelerator = "tpu"
        devices = args.tpu_cores
    else:
        accelerator = "gpu"
        devices = args.devices

    model_basename = os.path.basename(args.model)               # e.g. "Fender_deluxe_reverb.ckpt"
    model_name     = os.path.splitext(model_basename)[0]       # e.g. "Fender_deluxe_reverb"    

    wandb_logger = DynamicWandbLogger(
        project="ongsanet-training",
        entity="test_12342"
        # name="your_run_name",   # optional
        # log_model=True,         # optional
        # save_code=True          # optional
    )

    wandb_logger.experiment.config.update(vars(args))
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("models", model_name),
        filename=model_name + "-{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=1
    )


    # spectrogram_cb = SpectrogramCallback(every_n_epochs=1000, output_dir='./spectrograms')

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
    )

    ckpt_path = args.model if args.resume else None
    trainer.fit(model, ckpt_path=ckpt_path)
    trainer.save_checkpoint(args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", nargs="?", default="D:/Desktop/Archives/FIBO/Year3_2/opentopic/gtrin.wav")
    parser.add_argument("out_file", nargs="?", default="D:/Desktop/Archives/FIBO/Year3_2/opentopic/gtrout.wav")
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("--normalize", type=bool, default=True)

    parser.add_argument("--num_channels", type=int, default=18) #18 #4
    parser.add_argument("--dilation_depth", type=int, default=9) #16 #9
    parser.add_argument("--num_repeat", type=int, default=2) #3 #2
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use (e.g., GPUs)")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--tpu_cores", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--model", type=str, default="models/test/test.ckpt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
