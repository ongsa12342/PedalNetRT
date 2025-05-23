import argparse
import numpy as np
import json
import torch
import os

from model import PedalNet

def convert(args):
    """
    Converts a *.ckpt model from PedalNet into a .json format used in WaveNetVA.
    """

    a, b, c = (2, 1, 0)
    model = PedalNet.load_from_checkpoint(checkpoint_path=args.model)
    sd = model.state_dict()

    hparams = model.hparams
    residual_channels = hparams.num_channels
    filter_width = hparams.kernel_size
    dilations = [2 ** d for d in range(hparams.dilation_depth)] * hparams.num_repeat

    data_out = {
        "activation": "gated",
        "output_channels": 1,
        "input_channels": 1,
        "residual_channels": residual_channels,
        "filter_width": filter_width,
        "dilations": dilations,
        "variables": [],
    }

    # Use pytorch model data to populate the json data for each layer
    for i in range(-1, len(dilations) + 1):
        # Input Layer
        if i == -1:
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(w) for w in sd["wavenet.input_layer.weight"]
                                    .cpu()
                                    .permute(a, b, c)
                                    .flatten()
                                    .numpy()
                                    .tolist()
                    ],
                    "name": "W",
                }
            )
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(bias) for bias in sd["wavenet.input_layer.bias"]
                                     .cpu()
                                     .flatten()
                                     .numpy()
                                     .tolist()
                    ],
                    "name": "b",
                }
            )
        # Linear Mix Layer
        elif i == len(dilations):
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(w) for w in sd["wavenet.linear_mix.weight"]
                                    .cpu()
                                    .permute(a, b, c)
                                    .flatten()
                                    .numpy()
                                    .tolist()
                    ],
                    "name": "W",
                }
            )

            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(bias) for bias in sd["wavenet.linear_mix.bias"]
                                     .cpu()
                                     .numpy()
                                     .tolist()
                    ],
                    "name": "b",
                }
            )
        # Hidden Layers
        else:
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(w)
                        for w in sd[f"wavenet.hidden.{i}.weight"]
                                   .cpu()
                                   .permute(a, b, c)
                                   .flatten()
                                   .numpy()
                                   .tolist()
                    ],
                    "name": "W_conv",
                }
            )
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(bias)
                        for bias in sd[f"wavenet.hidden.{i}.bias"]
                                     .cpu()
                                     .flatten()
                                     .numpy()
                                     .tolist()
                    ],
                    "name": "b_conv",
                }
            )
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(w2)
                        for w2 in sd[f"wavenet.residuals.{i}.weight"]
                                   .cpu()
                                   .permute(a, b, c)
                                   .flatten()
                                   .numpy()
                                   .tolist()
                    ],
                    "name": "W_out",
                }
            )
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(b2)
                        for b2 in sd[f"wavenet.residuals.{i}.bias"]
                                   .cpu()
                                   .flatten()
                                   .numpy()
                                   .tolist()
                    ],
                    "name": "b_out",
                }
            )

    # Save final dictionary to a .json file
    out_file = args.model.replace(".ckpt", ".json")
    with open(out_file, "w") as outfile:
        json.dump(data_out, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
        default="models\models\Fender_Bassman50_Head\Fender_Bassman50_Head\models\Fender_Bassman50_Head\Fender_Bassman50_Head-epoch=1999.ckpt"
    )
    parser.add_argument("--format", default="json")
    args = parser.parse_args()
    convert(args)
