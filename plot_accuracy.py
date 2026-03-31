#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def extract_tensorboard_scalars(log_dir, tags):
    """
    Extract scalar data from TensorBoard logs

    Args:
        log_dir: path to tensorboard log directory (e.g., 'runs/...')
        tags: list of tag names to extract (e.g., ['Accuracy/train %', 'Accuracy/val %'])

    Returns:
        dict: {tag: [(step, value), ...]}
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    data = {}
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
        else:
            print(f"Warning: tag '{tag}' not found in {log_dir}")
            data[tag] = []

    return data


def plot_accuracy_scenario_a(log_dir='runs/scenario_a'):
    """
    Plot accuracy for Scenario A (key-reuse)
    """
    # Extract data
    tags = ['Accuracy/train %', 'Accuracy/val %']
    data = extract_tensorboard_scalars(log_dir, tags)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot training accuracy
    if data['Accuracy/train %']:
        steps, values = zip(*data['Accuracy/train %'])
        plt.plot(steps, values, linewidth=2, color='#1f77b4', label='Training')

    # Plot validation accuracy
    if data['Accuracy/val %']:
        steps, values = zip(*data['Accuracy/val %'])
        plt.plot(steps, values, linewidth=2, color='#ff7f0e', label='Validation')

    # Styling
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Scenario A: Key-Reuse with Random Split', fontsize=14, pad=15)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(left=1)
    plt.ylim(0, 105)

    # Save
    plt.tight_layout()
    plt.savefig('results/val_accuracy_scenario_a.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/val_accuracy_scenario_a.pdf")


def plot_accuracy_scenario_b(log_dir='runs/scenario_b'):
    """
    Plot accuracy for Scenario B (key-disjoint)
    """
    # Extract data
    tags = ['Accuracy/train %', 'Accuracy/val %']
    data = extract_tensorboard_scalars(log_dir, tags)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot training accuracy
    if data['Accuracy/train %']:
        steps, values = zip(*data['Accuracy/train %'])
        plt.plot(steps, values, linewidth=2, color='#1f77b4', label='Training')

    # Plot validation accuracy
    if data['Accuracy/val %']:
        steps, values = zip(*data['Accuracy/val %'])
        plt.plot(steps, values, linewidth=2, color='#ff7f0e', label='Validation')

        # Add horizontal line at 25% (random baseline)
        plt.axhline(y=25.0, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.7, label='Random Baseline (25%)')

    # Styling
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Scenario B: Key-Disjoint Split', fontsize=14, pad=15)
    plt.legend(fontsize=11, loc='right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(left=1)
    plt.ylim(0, 105)

    # Save
    plt.tight_layout()
    plt.savefig('results/val_accuracy_scenario_b.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/val_accuracy_scenario_b.pdf")


def plot_loss_scenario_a(log_dir='runs/Feb27_16-48-42_Maliks-MacBook-Pro.local'):
    """
    Plot loss for Scenario A
    """
    tags = ['Loss/train', 'Loss/val']
    data = extract_tensorboard_scalars(log_dir, tags)

    plt.figure(figsize=(10, 6))

    if data['Loss/train']:
        steps, values = zip(*data['Loss/train'])
        plt.plot(steps, values, linewidth=2, color='#1f77b4', label='Training')

    if data['Loss/val']:
        steps, values = zip(*data['Loss/val'])
        plt.plot(steps, values, linewidth=2, color='#ff7f0e', label='Validation')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Scenario A: Training and Validation Loss', fontsize=14, pad=15)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(left=1)

    plt.tight_layout()
    plt.savefig('results/val_loss_scenario_a.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/val_loss_scenario_a.pdf")


def plot_loss_scenario_b(log_dir='runs/scenario_b'):
    """
    Plot loss for Scenario B
    """
    tags = ['Loss/train', 'Loss/val']
    data = extract_tensorboard_scalars(log_dir, tags)

    plt.figure(figsize=(10, 6))

    if data['Loss/train']:
        steps, values = zip(*data['Loss/train'])
        plt.plot(steps, values, linewidth=2, color='#1f77b4', label='Training')

    if data['Loss/val']:
        steps, values = zip(*data['Loss/val'])
        plt.plot(steps, values, linewidth=2, color='#ff7f0e', label='Validation')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Scenario B: Training and Validation Loss', fontsize=14, pad=15)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(left=1)

    plt.tight_layout()
    plt.savefig('results/val_loss_scenario_b.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/val_loss_scenario_b.pdf")


if __name__ == '__main__':
    import sys

    # Detect tensorboard log directories
    runs_dir = 'runs'

    if not os.path.exists(runs_dir):
        print(f"Error: {runs_dir} directory not found!")
        sys.exit(1)

    # Find all subdirectories in runs/
    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
               if os.path.isdir(os.path.join(runs_dir, d))]

    if not subdirs:
        print(f"Error: No subdirectories found in {runs_dir}")
        sys.exit(1)

    print("Available log directories:")
    for i, d in enumerate(subdirs):
        print(f"  {i}: {d}")

    # Create dirs
    if not os.path.isdir("results"):
        os.makedirs("results")

    # Scenario A
    print("\n" + "=" * 60)
    print("Generating plots for Scenario A...")
    print("=" * 60)

    scenario_a_dir = [d for d in subdirs if 'scenario_b' not in d.lower()]
    if scenario_a_dir:
        log_dir_a = scenario_a_dir[0]
        print(f"Using: {log_dir_a}")
        plot_accuracy_scenario_a(log_dir_a)
        plot_loss_scenario_a(log_dir_a)
    else:
        print("Warning: No Scenario A log directory found")

    # Scenario B
    print("\n" + "=" * 60)
    print("Generating plots for Scenario B...")
    print("=" * 60)

    scenario_b_dir = [d for d in subdirs if 'scenario_b' in d.lower()]
    if scenario_b_dir:
        log_dir_b = scenario_b_dir[0]
        print(f"Using: {log_dir_b}")
        plot_accuracy_scenario_b(log_dir_b)
        plot_loss_scenario_b(log_dir_b)
    else:
        print("Warning: No Scenario B log directory found")

    print("\n" + "=" * 60)
    print("Done! Generated PDF files:")
    print("  - results/val_accuracy_scenario_a.pdf")
    print("  - results/val_loss_scenario_a.pdf")
    print("  - results/val_accuracy_scenario_b.pdf")
    print("  - results/val_loss_scenario_b.pdf")
    print("=" * 60)