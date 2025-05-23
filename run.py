import os
import sys
import subprocess

dataset_folder = "Dataset"
model_folder = "models"
in_file = "Dataset/AD_DA_converter.wav"  # your constant input file is inside "Dataset"

# Create the model folder if it doesn't exist
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Gather .wav files
wav_files = [f for f in os.listdir(dataset_folder) if f.endswith(".wav")]


wav_files = [
        "VOXac30_custom.wav"
    ]

for filename in wav_files:
    # Skip if it matches our in_file
    if filename == os.path.basename(in_file):
        print(f"Skipping file '{filename}' because it's the input file.")
        continue

    out_file = os.path.join(dataset_folder, filename)
    checkpoint_file = os.path.join(
        model_folder,
        os.path.splitext(filename)[0] + "2",
        os.path.splitext(filename)[0] + "2.ckpt"
    )

    # Note we now pass the input wav and output wav as *positional arguments*,
    # and we use `--model` for the checkpoint because train.py shows `--model MODEL` usage.
    command = [
        sys.executable,       # or just "python" if you prefer
        "train.py",
        in_file,              # positional argument 1
        out_file,             # positional argument 2
        "--model", checkpoint_file  # an optional argument recognized by train.py
    ]

    print(f"Processing '{out_file}' -> saving model to '{checkpoint_file}'")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running train.py on '{out_file}': {e}")
        print("Continuing with next file...\n")

print("All done!")
