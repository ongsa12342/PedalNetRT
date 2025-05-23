import subprocess

def run_predictions():
    # List of model names you want to process
    models = [
        "Roland_Jazz_Chorus",
        "VOXac30_custom"
    ]

    for model in models:
        # Construct file paths using the current model name
        in_file = f"Dataset/{model}.wav"
        out_file = f"Predicted/{model}.wav"
        model_path = f"models\\{model}\\{model}.ckpt"  # Windows-style path

        # Build the command string
        cmd = f'python predicting.py --in_file "{in_file}" --out_file "{out_file}" --model "{model_path}"'
        print(f"Executing command for model {model}:")
        print(cmd)
        
        # Execute the command
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    run_predictions()
