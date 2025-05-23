import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def plot_and_save_combined_spectrogram(file_dataset, file_pred, save_path, duration=None):
    """
    Loads two audio files (one from Dataset and one from Predicted), computes their spectrograms,
    and saves a single image with both spectrograms plotted in two subplots.
    
    Parameters:
        file_dataset (str): Path to the Dataset audio file.
        file_pred (str): Path to the Predicted audio file.
        save_path (str): Where to save the combined image.
        duration (float): Optional; limit to the first 'duration' seconds of the audio.
    """
    # Load audio (optionally limit duration to reduce memory usage)
    y_dataset, sr_dataset = librosa.load(file_dataset, sr=None, duration=duration)
    y_pred, sr_pred = librosa.load(file_pred, sr=None, duration=duration)
    
    # Check if sampling rates match
    if sr_dataset != sr_pred:
        print("Warning: Sampling rates differ between the two files!")
    sr = sr_pred  # Use the predicted file's sampling rate

    # Compute the spectrograms using a smaller FFT window to reduce memory usage.
    n_fft = 1024
    hop_length = 512

    D_dataset = np.abs(librosa.stft(y_dataset, n_fft=n_fft, hop_length=hop_length))
    D_db_dataset = librosa.amplitude_to_db(D_dataset, ref=np.max)

    D_pred = np.abs(librosa.stft(y_pred, n_fft=n_fft, hop_length=hop_length))
    D_db_pred = librosa.amplitude_to_db(D_pred, ref=np.max)

    # Create a figure with two subplots
    plt.figure(figsize=(14, 8))
    
    # Top subplot: Dataset spectrogram
    ax1 = plt.subplot(2, 1, 1)
    img1 = librosa.display.specshow(D_db_dataset, sr=sr, x_axis='time', y_axis='log', ax=ax1)
    plt.colorbar(img1, format='%+2.0f dB', ax=ax1)
    ax1.set_title(f"Spectrogram (Dataset): {os.path.basename(file_dataset)}")
    
    # Bottom subplot: Predicted spectrogram
    ax2 = plt.subplot(2, 1, 2)
    img2 = librosa.display.specshow(D_db_pred, sr=sr, x_axis='time', y_axis='log', ax=ax2)
    plt.colorbar(img2, format='%+2.0f dB', ax=ax2)
    ax2.set_title(f"Spectrogram (Predicted): {os.path.basename(file_pred)}")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Example usage:
models = [
    "Fender_Bassman50_Head",
    "Fender_deluxe_reverb",
    "Marshall_1959Plexi",
    "Marshall_JCM800",
    "Orange_Rocker30_Head",
    "Roland_Jazz_Chorus",
    "VOXac30_custom"
]

os.makedirs("Output/CombinedSpectrograms", exist_ok=True)

for model in models:
    file_dataset = rf"Dataset/{model}.wav"
    file_pred = rf"Predicted/{model}.wav"
    combined_save_path = rf"Output/CombinedSpectrograms/{model}_combined_spectrogram.png"
    plot_and_save_combined_spectrogram(file_dataset, file_pred, combined_save_path, duration=30)
    print(f"Saved combined spectrogram image for {model}")
