# sweep_entry.py
import yaml
import argparse
import wandb
from train import main  # your existing train.py entrypoint

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run():
    # 1) Initialize W&B (this populates wandb.config)
    wandb.init()

    # 2) Load the base template
    cfg = load_yaml(wandb.config.cfg_template)

    # 3) Override with the sweep’s hyperparameters
    cfg["learning_rate"] = wandb.config.learning_rate
    cfg["batch_size"]    = wandb.config.batch_size

    # 4) Turn into an argparse namespace and hand off to your main()
    args = argparse.Namespace(**cfg)
    main(args)

if __name__ == "__main__":
    run()
