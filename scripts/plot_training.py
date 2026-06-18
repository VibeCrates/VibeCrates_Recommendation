"""
Visualization script for training curves.
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# Previous training results (Session 11, before early stopping / regularization changes)
# Source: reports/report_session_12.txt
_PREV_STAGE1_VAL = [1.1347, 0.9800, 0.9700, 0.9700, 0.9700, 0.9700, 0.9700, 0.9700, 0.9700, 0.9674]
_PREV_STAGE2_VAL = [
    0.000365, 0.000362, 0.000359, 0.000357, 0.000355, 0.000354, 0.000353,
    0.000352, 0.000351, 0.000350, 0.000350, 0.000349, 0.000349, 0.000348, 0.000348,
]


def _epoch_averages(epochs, losses):
    """Compute per-epoch average loss from step-level arrays."""
    buckets = defaultdict(list)
    for ep, loss in zip(epochs, losses):
        buckets[ep].append(loss)
    sorted_epochs = sorted(buckets)
    avgs = [float(np.mean(buckets[e])) for e in sorted_epochs]
    return sorted_epochs, avgs


def plot_training_history(history_file: str, output_prefix: str = None):
    with open(history_file, 'r') as f:
        history = json.load(f)

    s1 = history['stage1']
    s2 = history['stage2']

    s1_epochs, s1_train = _epoch_averages(s1['epoch'], s1['total_loss'])
    s1_val = s1['val_loss']

    s2_epochs, s2_train = _epoch_averages(s2['epoch'], s2['loss'])
    s2_val = s2['val_loss']

    # ── Stage 1 graph ────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Stage 1: Contrastive Learning (Current Run)', fontsize=14, fontweight='bold')

    ax = axes1[0]
    ax.plot([e + 1 for e in s1_epochs], s1_train, 'b-o', label='Train Loss (avg)')
    ax.plot(range(1, len(s1_val) + 1), s1_val, 'r-o', label='Val Loss')
    best_ep = int(np.argmin(s1_val)) + 1
    ax.axvline(best_ep, color='green', linestyle='--', alpha=0.6, label=f'Best (epoch {best_ep})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss + Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes1[1]
    ax.plot([e + 1 for e in s1_epochs], s1['loss_tq'][:len(s1_epochs)] if len(s1['loss_tq']) >= len(s1_epochs) else
            [np.mean([s1['loss_tq'][i] for i, ep in enumerate(s1['epoch']) if ep == e]) for e in s1_epochs],
            label='TQ (Text↔Query)')
    # Use epoch averages for individual losses too
    _, s1_ti = _epoch_averages(s1['epoch'], s1['loss_ti'])
    _, s1_iq = _epoch_averages(s1['epoch'], s1['loss_iq'])
    _, s1_tq = _epoch_averages(s1['epoch'], s1['loss_tq'])
    ax.clear()
    ax.plot([e + 1 for e in s1_epochs], s1_tq, label='TQ (Text↔Query)', marker='o')
    ax.plot([e + 1 for e in s1_epochs], s1_ti, label='TI (Text↔Image)', marker='s')
    ax.plot([e + 1 for e in s1_epochs], s1_iq, label='IQ (Image↔Query)', marker='^')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Individual Contrastive Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig1, output_prefix, 'stage1')

    # ── Stage 2 graph ────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Stage 2: Knowledge Distillation (Current Run)', fontsize=14, fontweight='bold')

    ax = axes2[0]
    ax.plot([e + 1 for e in s2_epochs], s2_train, 'g-o', label='Train Loss (avg)')
    ax.plot(range(1, len(s2_val) + 1), s2_val, 'r-o', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train / Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(range(1, len(s2_val) + 1), s2_val, 'r-o', label='Val Loss (zoom)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Val Loss Detail')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig2, output_prefix, 'stage2')

    # ── Comparison graph ──────────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle(
        'Comparison: Previous vs Current Run\n'
        '(Previous: no early-stop, dropout=0.05, wd=1e-5  |  '
        'Current: early-stop patience=3, dropout=0.1, wd=1e-2)',
        fontsize=11, fontweight='bold'
    )

    ax = axes3[0]
    ax.plot(range(1, len(_PREV_STAGE1_VAL) + 1), _PREV_STAGE1_VAL,
            'b--o', alpha=0.7, label='Prev Val Loss (10 ep, no ES)')
    ax.plot(range(1, len(s1_val) + 1), s1_val,
            'r-o', label=f'Curr Val Loss ({len(s1_val)} ep, early-stop)')
    best_prev = min(_PREV_STAGE1_VAL)
    best_curr = min(s1_val)
    ax.axhline(best_prev, color='blue', linestyle=':', alpha=0.5, label=f'Prev best {best_prev:.4f}')
    ax.axhline(best_curr, color='red', linestyle=':', alpha=0.5, label=f'Curr best {best_curr:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss')
    ax.set_title('Stage 1 — Val Loss Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    ax.plot(range(1, len(_PREV_STAGE2_VAL) + 1), _PREV_STAGE2_VAL,
            'b--o', alpha=0.7, label='Prev Val Loss')
    ax.plot(range(1, len(s2_val) + 1), s2_val,
            'r-o', label='Curr Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss')
    ax.set_title('Stage 2 — Val Loss Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig3, output_prefix, 'comparison')

    plt.close('all')

    # Print summary
    print("\n=== Training Summary ===")
    print(f"Stage 1 — {len(s1_val)} epochs (early stopping at patience=3)")
    print(f"  Train Loss : {s1_train[0]:.4f} → {s1_train[-1]:.4f}")
    print(f"  Val Loss   : {s1_val[0]:.4f} → {s1_val[-1]:.4f}  (best: {min(s1_val):.4f} @ epoch {int(np.argmin(s1_val))+1})")
    print(f"  Prev best  : {min(_PREV_STAGE1_VAL):.4f}  →  Δ = {min(_PREV_STAGE1_VAL) - min(s1_val):+.4f}")
    print()
    print(f"Stage 2 — {len(s2_val)} epochs")
    print(f"  Train Loss : {s2_train[0]:.6f} → {s2_train[-1]:.6f}")
    print(f"  Val Loss   : {s2_val[0]:.6f} → {s2_val[-1]:.6f}")
    print(f"  Prev final : {_PREV_STAGE2_VAL[-1]:.6f}  →  Δ = {_PREV_STAGE2_VAL[-1] - s2_val[-1]:+.6f}")


def _save(fig, prefix, tag):
    if prefix:
        path = Path(prefix).parent / f"{Path(prefix).name}_{tag}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--history', type=str, required=True)
    parser.add_argument('--output', type=str, default=None,
                        help='Output prefix, e.g. assets/training_history_20260618')
    args = parser.parse_args()
    plot_training_history(args.history, args.output)
