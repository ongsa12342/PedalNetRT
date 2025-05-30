import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
import pickle
import os

import time

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self._padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = torch.nn.functional.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        if self._padding != 0:
            return result[:, :, : -self._padding]
        return result


def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )


class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        internal_channels = int(num_channels * 2)
        self.hidden = _conv_stack(dilations, num_channels, internal_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)
        self.input_layer = CausalConv1d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=1,
        )

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
        )
        self.num_channels = num_channels

    def forward(self, x):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # gated activation
            #   split (32,16,3) into two (16,16,3) for tanh and sigm calculations
            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out


def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)


def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class PedalNet(pl.LightningModule):
    def __init__(self, hparams):
        super(PedalNet, self).__init__()

        torch.set_float32_matmul_precision('high')

        self.training_step_outputs = []

        self.save_hyperparameters(hparams)

        self.wavenet = WaveNet(
            num_channels=hparams["num_channels"],
            dilation_depth=hparams["dilation_depth"],
            num_repeat=hparams["num_repeat"],
            kernel_size=hparams["kernel_size"],
        )
        # self.hparams = hparams

    def prepare_data(self):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data = pickle.load(open(os.path.dirname(self.hparams.model) + "/data.pickle", "rb"))
        self.train_ds = ds(data["x_train"], data["y_train"])
        self.valid_ds = ds(data["x_valid"], data["y_valid"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.wavenet.parameters(), lr=self.hparams.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-6
        )
        return {
       "optimizer": optimizer,
       "lr_scheduler": {
           "scheduler": scheduler,
           "monitor": "val_loss"
       }
    }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            # persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size, num_workers=0,  pin_memory=True)

    def forward(self, x):
        return self.wavenet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2):], y_pred).mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def on_train_epoch_start(self):
        self.epoch_start = time.time()

    def on_train_epoch_end(self):
        
        epoch_time = time.time() - self.epoch_start
        samples = len(self.train_ds) * self.trainer.accumulate_grad_batches
        self.log("samples_per_sec", samples / epoch_time, on_epoch=True)
        self.log("epoch_time_s", epoch_time, on_epoch=True)

        avg_loss = torch.stack(self.training_step_outputs).mean()
        
        self.log("avg_train_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.clear()
