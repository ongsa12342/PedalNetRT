# summarize_sweep.py
import wandb
import pandas as pd

# 1) Point to your sweep
ENTITY  = "test_12342"
PROJECT = "ongsanet-training"
SWEEP_ID = "v35wzlq5"   # replace with your sweep id

api   = wandb.Api()
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")

# 2) Gather all runs into a DataFrame
records = []
for run in sweep.runs:
    # skip unfinished or errored runs
    if run.state != "finished":
        continue

    # read config & final metric
    cfg = run.config
    summary = run.summary
    records.append({
        "batch_size":      cfg.get("batch_size"),
        "learning_rate":   cfg.get("learning_rate"),
        "val_loss":        summary.get("val_loss"),
        "steps":           summary.get("global_step"),  # or summary["_step"]
        "duration_sec":    summary.get("epoch_time_s") * summary.get("epoch") if summary.get("epoch_time_s") else None
    })

df = pd.DataFrame(records)

# 3) Sort by val_loss and show top 5
print("🏆 Top 5 runs:")
print(df.sort_values("val_loss").head(5))

# 4) Aggregate statistics by batch_size or LR
print("\n🔍 Mean & std of val_loss by batch_size:")
print(df.groupby("batch_size")["val_loss"].agg(["mean","std"]))

print("\n🔍 Mean & std of val_loss by learning_rate:")
print(df.groupby("learning_rate")["val_loss"].agg(["mean","std"]))

# 5) (Optional) Pivot to see matrix of BS vs LR
pivot = df.pivot_table(
    index="batch_size",
    columns="learning_rate",
    values="val_loss",
    aggfunc="mean"
)
print("\n🗺️ Pivot table (mean val_loss):")
print(pivot)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    1,                       # one row …
    df["batch_size"].nunique(),  # … one column per distinct batch size
    figsize=(4 * df["batch_size"].nunique(), 4),
    sharey=True
)

if df["batch_size"].nunique() == 1:
    axes = [axes]   # make it iterable when there’s only one BS

for ax, (bs, grp) in zip(axes, df.groupby("batch_size")):
    ax.scatter(
        grp["learning_rate"],
        grp["val_loss"],
        alpha=0.7,
        s=50,
        label=f"bs={bs}"
    )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_title(f"batch_size = {bs}")
    ax.grid(True, which="both", ls="--", alpha=0.3)

axes[0].set_ylabel("val_loss")
plt.tight_layout()
plt.show()