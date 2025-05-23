import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import argparse
import os
from scipy import signal


def pre_emphasis_filter(x, coeff=0.95):
    """
    Applies a simple pre-emphasis filter to x
    """
    return np.concatenate([x, x - coeff * x])


def error_to_signal(y, y_pred, use_filter=1):
    """
    Error to signal ratio with optional pre-emphasis filter.
    Both y and y_pred are truncated to the shorter length.
    """
    # Truncate to the shorter length to avoid mismatched shapes
    min_len = min(len(y), len(y_pred))
    y = y[:min_len]
    y_pred = y_pred[:min_len]

    if use_filter == 1:
        y = pre_emphasis_filter(y)
        y_pred = pre_emphasis_filter(y_pred)

        # After pre-emphasis, y and y_pred are still guaranteed same length
        # (we used the same operation on both)
        # But if you'd like to be extra safe:
        # y, y_pred = y[:len(y_pred)], y_pred[:len(y)]

    return np.sum((y - y_pred) ** 2) / (np.sum(y**2) + 1e-10)



def read_wave(wav_file):
    """
    Reads a WAV file and returns (signal_data, sample_rate).
    """
    fs, signal_data = wavfile.read(wav_file)
    return signal_data, fs


def analyze_pred_vs_actual(args):
    """
    Generates plots to analyze the predicted signal vs. the actual signal.
    """

    # ---------------------------------------------------------------------
    # Collect file arguments
    # ---------------------------------------------------------------------
    input_wav = args.in_file  # e.g., "x_test.wav"
    out_wav = args.out_file   # e.g., "y_test.wav" (actual/target)
    pred_wav = args.pred_file # e.g., "y_pred.wav"

    # The output plots will go in the same folder as the model checkpoint
    model_dir = os.path.dirname(args.model)
    if not model_dir:
        # If user only gave a filename (no path), default to current dir
        model_dir = "."

    # ---------------------------------------------------------------------
    # 1) Read the input wav file (the "dry" or "pre-effect" signal)
    # ---------------------------------------------------------------------
    signal_in, fs_in = read_wave(input_wav)

    # 2) Read the actual output wav (the "wet" or target effect)
    signal_out, fs_out = read_wave(out_wav)
    time_out = np.linspace(0, len(signal_out) / fs_out, num=len(signal_out))

    # 3) Read the predicted wav (the model's attempt at the effect)
    signal_pred, fs_pred = read_wave(pred_wav)
    time_pred = np.linspace(0, len(signal_pred) / fs_pred, num=len(signal_pred))

    # ---------------------------------------------------------------------
    # Calculate error
    # ---------------------------------------------------------------------
    # Zip the shorter length to avoid out-of-bounds if signals differ slightly
    shorter_len = min(len(signal_out), len(signal_pred))
    error_list = np.abs(signal_pred[:shorter_len] - signal_out[:shorter_len])

    # Error to Signal ratio
    e2s = error_to_signal(signal_out, signal_pred)          # with pre-emphasis
    e2s_no_filter = error_to_signal(signal_out, signal_pred, use_filter=0)

    print(f"Error to signal (with pre-emphasis filter): {e2s}")
    print(f"Error to signal (no pre-emphasis filter):   {e2s_no_filter}")

    # ---------------------------------------------------------------------
    # Make comparison plots (3 subplots)
    # ---------------------------------------------------------------------
    fig, (ax_in, ax_comp, ax_err) = plt.subplots(3, 1, sharex=False, figsize=(13, 8))
    fig.suptitle(f"Predicted vs Actual Signal (Error-to-Signal: {round(e2s, 4)})")

    # (A) Plot the original input signal on ax_in
    time_in = np.linspace(0, len(signal_in) / fs_in, num=len(signal_in))
    ax_in.plot(time_in, signal_in, label=os.path.basename(input_wav))
    ax_in.set_title("Original Input Signal")
    ax_in.set_ylabel("Amplitude")
    ax_in.grid(True)
    ax_in.legend()

    # (B) Plot the actual and predicted signals on ax_comp
    ax_comp.plot(time_out, signal_out, label=os.path.basename(out_wav), alpha=0.7)
    ax_comp.plot(time_pred, signal_pred, label=os.path.basename(pred_wav), alpha=0.7)
    ax_comp.set_title("Target (Actual) vs Predicted")
    ax_comp.set_ylabel("Amplitude")
    ax_comp.grid(True)
    ax_comp.legend()

    # (C) Plot the absolute difference
    time_err = np.linspace(0, len(error_list) / fs_out, num=len(error_list))
    ax_err.plot(time_err, error_list, label="|pred - actual|")
    ax_err.set_title("Absolute Error over Time")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Error")
    ax_err.grid(True)
    ax_err.legend()

    # ---------------------------------------------------------------------
    # Save main figure
    # ---------------------------------------------------------------------
    # e.g., "<model_dir>/signal_comparison_e2s_0.1234.png"
    main_fig_path = os.path.join(
        model_dir, f"signal_comparison_e2s_{round(e2s, 4)}.png"
    )
    plt.savefig(main_fig_path, bbox_inches="tight")
    print(f"Saved main comparison plot to: {main_fig_path}")

    # ---------------------------------------------------------------------
    # Create a zoomed-in figure around the max amplitude of the target signal
    # ---------------------------------------------------------------------
    # (This next part can fail if the target signal is silent. So we wrap it.)
    try:
        max_index = np.argmax(np.abs(signal_pred))
        # e.g., ~10ms window around that point
        start_time = max_index / fs_out - 0.005
        end_time = max_index / fs_out + 0.005

        # Reuse the same figure or create a new one
        plt.figure(figsize=(13, 6))
        plt.plot(time_out, signal_out, label="target: "+os.path.basename(out_wav))
        plt.plot(time_pred, signal_pred, label="predicted: "+os.path.basename(pred_wav))
        plt.title("Zoomed-In: Target vs Predicted")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.axis([start_time, end_time, min(signal_out), max(signal_out)])
        zoom_fig_path = os.path.join(
            model_dir, f"detail_signal_comparison_e2s_{round(e2s, 4)}.png"
        )
        plt.savefig(zoom_fig_path, bbox_inches="tight")
        print(f"Saved zoomed-in comparison plot to: {zoom_fig_path}")
    except ValueError:
        print("Warning: Could not compute zoomed plot (signal_out might be empty).")

    # ---------------------------------------------------------------------
    # Plot the spectrogram of the difference
    # ---------------------------------------------------------------------
    diff_signal = signal_pred[:shorter_len] - signal_out[:shorter_len]
    plt.figure(figsize=(12, 8))
    frequencies, times, spectrogram_data = signal.spectrogram(diff_signal, fs_out)
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data + 1e-14))
    plt.title("Diff Spectrogram (Predicted - Actual)")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Power (dB)")

    spectro_path = os.path.join(model_dir, "diff_spectrogram.png")
    plt.savefig(spectro_path, bbox_inches="tight")
    print(f"Saved difference spectrogram to: {spectro_path}")

    # If user passed --show, display the plots
    if args.show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze predicted vs. actual .wav files."
    )
    # Three wave files as arguments
    parser.add_argument(
        "--in_file",
        required=True,
        help="Path to the 'input' or dry .wav file"
    )
    parser.add_argument(
        "--out_file",
        required=True,
        help="Path to the 'output' or actual/target .wav file"
    )
    parser.add_argument(
        "--pred_file",
        required=True,
        help="Path to the predicted .wav file"
    )
    parser.add_argument(
        "--model",
        default="models/pedalnet/ppedalnet.ckpt",
        help="Path to your checkpoint file (only used for naming plot files)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (defaults to saving only)."
    )
    args = parser.parse_args()

    analyze_pred_vs_actual(args)
