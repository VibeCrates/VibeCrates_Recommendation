import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

with open("logs/training_history.json") as f:
    data = json.load(f)

s1 = data["stage1"]
s2 = data["stage2"]

# ── epoch-level averages for step-wise data ──────────────────────────────────
def epoch_means(epoch_list, value_list):
    epochs = sorted(set(epoch_list))
    means = []
    for e in epochs:
        vals = [v for ep, v in zip(epoch_list, value_list) if ep == e]
        means.append(np.mean(vals))
    return epochs, means

s1_epochs, s1_total   = epoch_means(s1["epoch"], s1["total_loss"])
_,          s1_tq     = epoch_means(s1["epoch"], s1["loss_tq"])
_,          s1_ti     = epoch_means(s1["epoch"], s1["loss_ti"])
_,          s1_iq     = epoch_means(s1["epoch"], s1["loss_iq"])

s2_epochs, s2_train   = epoch_means(s2["epoch"], s2["loss"])

# val_loss indices start at epoch 1
s1_val_epochs = list(range(1, len(s1["val_loss"]) + 1))
s2_val_epochs = list(range(1, len(s2["val_loss"]) + 1))

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"

# ── Stage 1 figure ────────────────────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle("Stage 1 – Contrastive Pre-training", fontsize=13, fontweight="bold", y=1.01)

ax1.plot(s1_epochs, s1_total, color=BLUE,   marker="o", ms=4, label="Train (total)")
ax1.plot(s1_val_epochs, s1["val_loss"], color=ORANGE, marker="s", ms=5,
         linestyle="--", label="Val")
ax1.set_title("Total Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.plot(s1_epochs, s1_tq, color=BLUE,  marker="o", ms=3, label="Text↔Query (tq)")
ax2.plot(s1_epochs, s1_ti, color=GREEN, marker="s", ms=3, label="Text↔Image (ti)")
ax2.plot(s1_epochs, s1_iq, color=RED,   marker="^", ms=3, label="Image↔Query (iq)")
ax2.set_title("Component Losses")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

fig1.tight_layout()
out1 = "assets/training_history_stage1.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved → {out1}")

# ── Stage 2 figure ────────────────────────────────────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Stage 2 – Knowledge Distillation", fontsize=13, fontweight="bold", y=1.01)

ax3.plot(s2_epochs, s2_train, color=BLUE,   marker="o", ms=4, label="Train")
ax3.plot(s2_val_epochs, s2["val_loss"], color=ORANGE, marker="s", ms=5,
         linestyle="--", label="Val")
ax3.set_title("Train & Val Loss")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

ax4.plot(s2_val_epochs, s2["val_loss"], color=ORANGE, marker="s", ms=5,
         linestyle="--", label="Val Loss")
ax4.set_title("Val Loss (detail)")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Loss")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
final_val = s2["val_loss"][-1]
ax4.annotate(f"{final_val:.6f}", xy=(s2_val_epochs[-1], final_val),
             xytext=(-45, 10), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

fig2.tight_layout()
out2 = "assets/training_history_stage2.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
