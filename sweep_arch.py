#!/usr/bin/env python
"""
sweep_arch.py – W&B agent launcher

Merges a base YAML config with sweep params, then calls train.py by
expanding every key/value into CLI flags.  
Boolean handling:
  • keys in BOOL_FLAGS are treated as store_true/false flags →
    emit flag only when value=True.  
  • all other booleans are passed as explicit values ("true"/"false").
Positional args (in_file, out_file) are appended without flags.
"""

import argparse, os, subprocess, yaml, wandb

POSITIONALS = ("in_file", "out_file")
BOOL_FLAGS  = {"cpu", "resume"}   # store_true/false style flags in train.py

def parse_args():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--cfg_template", required=True)
    args, _ = p.parse_known_args()
    return args


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args      = parse_args()
    run       = wandb.init()
    sweep_cfg = dict(run.config)

    base = load_yaml(args.cfg_template)
    base.update({k: v for k, v in sweep_cfg.items() if k != "cfg_template"})

    cmd = ["python", "train.py"]

    for k, v in base.items():
        if k in POSITIONALS or v is None:
            continue

        if k in BOOL_FLAGS:
            # emit flag only when True
            if v:
                cmd.append(f"--{k}")
        else:
            # regular named arg
            val = str(v).lower() if isinstance(v, bool) else str(v)
            cmd += [f"--{k}", val]

    # positional args
    cmd += [base[p] for p in POSITIONALS]

    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
