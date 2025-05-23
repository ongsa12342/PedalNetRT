# predicting.py

import argparse
import os
import pickle
import torch
import numpy as np
from scipy.io import wavfile
from model import PedalNet  # Imports PedalNet and WaveNet from model.py


def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max, abs(data_min))
    return data / (data_norm if data_norm != 0 else 1.0)


def load_model_and_data(model_ckpt_path):
    """
    Loads the trained PedalNet model from a checkpoint and the associated mean/std 
    from data.pickle for proper standardization/de-standardization.
    """
    # 1) Load hyperparameters from checkpoint
    #    Set weights_only=True to suppress the future warning about untrusted model files
    ckpt = torch.load(model_ckpt_path, map_location='cpu', weights_only=False)
    hparams = ckpt["hyper_parameters"]

    # 2) Initialize PedalNet with the same hyperparameters
    model = PedalNet(hparams)

    # 3) Load model state
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 4) Load mean and std from the stored data
    data_pickle_path = os.path.join(os.path.dirname(model_ckpt_path), "data.pickle")
    if not os.path.exists(data_pickle_path):
        raise FileNotFoundError(
            f"Could not find {data_pickle_path}. Make sure it exists in the same folder as the checkpoint."
        )

    with open(data_pickle_path, "rb") as f:
        data_dict = pickle.load(f)

    mean_val = data_dict["mean"]
    std_val = data_dict["std"]

    return model, mean_val, std_val, hparams


def process_audio(audio, model, mean_val, std_val, sample_rate, sample_time, device):
    """
    Takes raw audio (numpy array), breaks it into chunks,
    standardizes, predicts through the model, and returns output audio.
    """

    # 1) Optional normalization (performed in training).
    audio = normalize(audio)

    # 2) Reshape into (N, 1, sample_size) chunks
    sample_size = int(sample_rate * sample_time)
    length = len(audio) - len(audio) % sample_size
    audio_trimmed = audio[:length]
    num_chunks = len(audio_trimmed) // sample_size

    # 3) Standardize using training mean & std
    # audio_trimmed = (audio_trimmed - mean_val) / (std_val if std_val != 0 else 1.0)

    # 4) Convert to a PyTorch tensor of shape (batch, 1, sample_size)
    audio_tensor = torch.from_numpy(audio_trimmed).float().view(num_chunks, 1, sample_size)

    # 5) Predict in chunks
    with torch.no_grad():
        predictions = []
        for i in range(num_chunks):
            x_in = audio_tensor[i : i+1]  # shape (1, 1, sample_size)

            # Ensure your input is on the same device as the model
            x_in = x_in.to(device)

            y_pred = model(x_in)
            y_pred = y_pred.squeeze().cpu().numpy()  # move back to CPU for concatenation

            predictions.append(y_pred)

    # 6) Concatenate all chunk predictions
    pred_audio = np.concatenate(predictions, axis=0)

    # 7) De-standardize
    # pred_audio = (pred_audio * std_val) + mean_val

    # 8) (Optional) Re-normalize
    # pred_audio = normalize(pred_audio)

    return pred_audio


def main(args):
    # 1) Load the model, mean, std
    model, mean_val, std_val, hparams = load_model_and_data(args.model)

    device = "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    model.to(device)

    # 2) Read the input audio
    in_rate, in_data = wavfile.read(args.in_file)
    print(f"Loaded input file '{args.in_file}' at sample rate {in_rate}.")

    # 3) If stereo, use only the left channel
    if len(in_data.shape) > 1:
        print("[INFO] Stereo input detected. Using the first (left) channel only.")
        in_data = in_data[:, 0]

    # 4) Convert from int16 to float if needed
    if in_data.dtype == np.int16:
        in_data = in_data / 32767.0

    # 5) Process the audio through the model
    pred_audio = process_audio(
        in_data, model, mean_val, std_val, in_rate, args.sample_time, device
    )

    # 6) Write out the prediction to a wave file
    #    Scale back to int16 range if desired.
    if args.output_int16:
        out_wav = (pred_audio * 32767).astype(np.int16)
    else:
        # Using float32 for output
        out_wav = pred_audio.astype(np.float32)

    wavfile.write(args.out_file, in_rate, out_wav)
    print(f"Saved predicted output to '{args.out_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using the trained PedalNet model."
    )
    parser.add_argument("--in_file", type=str, required=True, help="Path to input WAV file.")
    parser.add_argument("--out_file", type=str, required=True, help="Path to output WAV file.")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (e.g. .ckpt).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for inference, even if GPU is available.")
    parser.add_argument("--sample_time", type=float, default=1.0,
                        help="Chunk size in seconds (should match training).")
    parser.add_argument("--output_int16", action="store_true",
                        help="If set, output WAV is written as 16-bit PCM. Otherwise float32.")
    args = parser.parse_args()
    main(args)
