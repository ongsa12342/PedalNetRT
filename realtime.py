import argparse
import os
import pickle
import time
import torch
import numpy as np
from scipy.io import wavfile
from model import PedalNet


def normalize(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data_norm = max(data_max, abs(data_min))
    return data / (data_norm if data_norm != 0 else 1.0)


def load_model_and_data(model_ckpt_path, weights_only=False):
    """
    Loads the trained PedalNet model from a checkpoint and the associated mean/std
    from data.pickle for proper standardization/de-standardization.
    """
    ckpt = torch.load(model_ckpt_path, map_location='cpu', weights_only=weights_only)
    hparams = ckpt["hyper_parameters"]

    model = PedalNet(hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

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


def simulate_realtime_inference(
    audio,
    model,
    mean_val,
    std_val,
    sample_rate,
    block_size,
    device,
    print_stats=True,
):
    """
    Simulates real-time processing by splitting 'audio' into small blocks (block_size)
    and passing each block to the model in sequence, measuring the processing time.

    Args:
        audio (np.array): 1D array of audio samples (float).
        model (nn.Module): Trained PedalNet model (on device).
        mean_val (float): Mean used in training set standardization.
        std_val (float): Std used in training set standardization.
        sample_rate (int): Sample rate of the audio.
        block_size (int): Number of samples per chunk. 
                          e.g. 256 for ~5.8ms at 44.1 kHz, or 512 for ~11.6ms, etc.
        device (str): "cpu" or "cuda".
        print_stats (bool): If True, prints real-time stats at the end.

    Returns:
        np.array: The full “processed” (inferred) audio, concatenated block by block.
    """

    # Optional: re-normalize to match training
    audio = normalize(audio)

    # Standardize entire audio (like your training).
    audio = (audio - mean_val) / (std_val if std_val != 0 else 1.0)

    # We'll simulate real-time block-based streaming
    num_blocks = len(audio) // block_size
    leftover = len(audio) % block_size

    # Container to hold final processed audio
    processed_audio = []

    # Keep track of timings
    times = []

    with torch.no_grad():
        for i in range(num_blocks):
            block = audio[i * block_size : (i + 1) * block_size]

            # Convert block to a shape of (batch=1, channels=1, time=block_size)
            block_tensor = torch.from_numpy(block).float().unsqueeze(0).unsqueeze(0).to(device)

            # Start timing
            t0 = time.perf_counter()

            # Run the model forward
            y_pred = model(block_tensor)

            # Stop timing
            t1 = time.perf_counter()

            # Measure time for this block
            times.append(t1 - t0)

            # Typically the model might produce fewer samples if there's a causal padding trim.
            # For simplicity, we assume it's the same length:
            # shape is (1, 1, block_size) or close to that. 
            y_pred = y_pred.squeeze().cpu().numpy()

            processed_audio.append(y_pred)

        # If there's leftover audio (less than one full block_size), process it as one final block
        if leftover > 0:
            block = audio[num_blocks * block_size :]

            block_tensor = torch.from_numpy(block).float().unsqueeze(0).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            y_pred = model(block_tensor)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            y_pred = y_pred.squeeze().cpu().numpy()
            processed_audio.append(y_pred)

    # Concatenate all processed blocks
    pred_audio = np.concatenate(processed_audio, axis=0)

    # De-standardize
    pred_audio = (pred_audio * std_val) + mean_val
    # Optional renormalization
    pred_audio = normalize(pred_audio)

    # Print some statistics
    if print_stats and len(times) > 0:
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        # Each block is block_size samples. Real-time means block_size / sample_rate seconds of audio
        # should be processed in less time than that block_size duration.
        block_duration_sec = block_size / sample_rate
        realtime_factor = block_duration_sec / avg_time
        # If realtime_factor >= 1.0, we are at or faster than real-time on average.
        print("------- Real-Time Simulation Stats -------")
        print(f"Number of blocks processed: {len(times)}")
        print(f"Block size: {block_size} samples "
              f"({block_duration_sec * 1000:.2f} ms of audio per block)")
        print(f"Average processing time per block: {avg_time*1000:.3f} ms")
        print(f"Min processing time: {min_time*1000:.3f} ms")
        print(f"Max processing time: {max_time*1000:.3f} ms")
        print(f"Real-time factor (block_duration / avg_time): {realtime_factor:.3f}")
        if realtime_factor >= 1.0:
            print(f"[OK] Average throughput meets or exceeds real-time.\n")
        else:
            print(f"[WARNING] Average throughput is below real-time.\n")

    return pred_audio


def main(args):
    """
    Main entry point: simulates real-time inference using the model,
    and saves the result as a WAV file. 
    """
    # 1) Load the model
    model, mean_val, std_val, hparams = load_model_and_data(args.model)

    # 2) Pick the device
    device = "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    model.to(device)

    # 3) Load the input file
    in_rate, in_data = wavfile.read(args.in_file)
    print(f"Loaded input file '{args.in_file}' at sample rate {in_rate}.")

    # If stereo, take left channel
    if len(in_data.shape) > 1:
        print("[INFO] Stereo input. Using only left channel.")
        in_data = in_data[:, 0]

    # Convert int16 -> float
    if in_data.dtype == np.int16:
        in_data = in_data.astype(np.float32) / 32767.0

    # 4) Simulate real-time inference
    pred_audio = simulate_realtime_inference(
        audio=in_data,
        model=model,
        mean_val=mean_val,
        std_val=std_val,
        sample_rate=in_rate,
        block_size=args.block_size,
        device=device,
        print_stats=True,
    )

    # 5) Save output
    if args.output_int16:
        out_wav = (pred_audio * 32767).astype(np.int16)
    else:
        out_wav = pred_audio.astype(np.float32)

    wavfile.write(args.out_file, in_rate, out_wav)
    print(f"Saved simulated real-time output to '{args.out_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate real-time inference and benchmark performance with PedalNet."
    )
    parser.add_argument("--in_file", type=str, required=True, help="Path to input WAV file.")
    parser.add_argument("--out_file", type=str, required=True, help="Path to output WAV file.")
    parser.add_argument("--model", type=str, required=True, help="Path to .ckpt model checkpoint.")
    parser.add_argument("--block_size", type=int, default=512,
                        help="Number of samples per block (e.g., 256, 512, 1024).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU for inference, even if GPU is available.")
    parser.add_argument("--output_int16", action="store_true",
                        help="Write WAV as 16-bit PCM instead of float32.")
    args = parser.parse_args()
    main(args)
