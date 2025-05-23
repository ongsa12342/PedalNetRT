import yaml
import argparse
import sys
import subprocess

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    command = [
        sys.executable, "train.py",
        config["in_file"],
        config["out_file"],
        "--model", config["model"],
        "--num_channels", str(config["num_channels"]),
        "--dilation_depth", str(config["dilation_depth"]),
        "--num_repeat", str(config["num_repeat"]),
        "--kernel_size", str(config["kernel_size"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--max_epochs", str(config["max_epochs"]),
        "--devices", str(config["devices"])
    ]

    if config.get("resume"):
        command.append("--resume")
    if config.get("cpu"):
        command.append("--cpu")

    subprocess.run(command)

if __name__ == "__main__":
    main()
