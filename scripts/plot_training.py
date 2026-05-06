"""
Visualization script for training curves.
"""
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_training_history(history_file: str, output_file: str = None):
    """
    Plot training curves from history file.
    
    Args:
        history_file: Path to history JSON file
        output_file: Path to save the plot (optional)
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Stage 1: Total Loss
    if history['stage1']['epoch']:
        ax = axes[0, 0]
        ax.plot(history['stage1']['epoch'], history['stage1']['total_loss'], 'b-', label='Train Loss', alpha=0.7)
        if history['stage1']['val_loss']:
            # Assuming one val_loss per epoch
            unique_epochs = sorted(set(history['stage1']['epoch']))
            ax.plot(unique_epochs[:len(history['stage1']['val_loss'])], history['stage1']['val_loss'], 'r-', label='Val Loss', marker='o')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Stage 1: Total Loss (Contrastive Learning)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Stage 1: Individual Losses
    if history['stage1']['epoch']:
        ax = axes[0, 1]
        ax.plot(history['stage1']['epoch'], history['stage1']['loss_tq'], label='Loss_TQ', alpha=0.7)
        ax.plot(history['stage1']['epoch'], history['stage1']['loss_ti'], label='Loss_TI', alpha=0.7)
        ax.plot(history['stage1']['epoch'], history['stage1']['loss_iq'], label='Loss_IQ', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Stage 1: Individual Contrastive Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Stage 2: KL-Divergence Loss
    if history['stage2']['epoch']:
        ax = axes[1, 0]
        ax.plot(history['stage2']['epoch'], history['stage2']['loss'], 'g-', label='Train Loss', alpha=0.7)
        if history['stage2']['val_loss']:
            unique_epochs = sorted(set(history['stage2']['epoch']))
            ax.plot(unique_epochs[:len(history['stage2']['val_loss'])], history['stage2']['val_loss'], 'r-', label='Val Loss', marker='o')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Stage 2: KL-Divergence Loss (Distillation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "Training Summary:\n\n"
    
    if history['stage1']['epoch']:
        avg_stage1_loss = sum(history['stage1']['total_loss']) / len(history['stage1']['total_loss'])
        summary_text += f"Stage 1 Avg Loss: {avg_stage1_loss:.4f}\n"
        summary_text += f"Stage 1 Batches: {len(history['stage1']['epoch'])}\n"
    
    if history['stage2']['epoch']:
        avg_stage2_loss = sum(history['stage2']['loss']) / len(history['stage2']['loss'])
        summary_text += f"Stage 2 Avg Loss: {avg_stage2_loss:.4f}\n"
        summary_text += f"Stage 2 Batches: {len(history['stage2']['epoch'])}\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument(
        '--history',
        type=str,
        required=True,
        help='Path to history JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the plot (optional, shows plot if not provided)'
    )
    
    args = parser.parse_args()
    plot_training_history(args.history, args.output)
