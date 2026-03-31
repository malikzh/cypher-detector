#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset, random_split
from dataset import Dataset
from model import CipherClassifier
from config import get_configuration
from loguru import logger as log
from os.path import abspath
import os


def get_current_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_key_disjoint_split(dataset, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset by key_id to ensure no key overlap
    """
    samples_by_key = {}
    for idx in range(len(dataset)):
        key_id = dataset.get_key_id(idx)
        if key_id not in samples_by_key:
            samples_by_key[key_id] = []
        samples_by_key[key_id].append(idx)

    unique_keys = sorted(samples_by_key.keys())
    np.random.seed(42)
    np.random.shuffle(unique_keys)

    n_keys = len(unique_keys)
    n_train = int(n_keys * train_ratio)
    n_val = int(n_keys * val_ratio)

    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train:n_train + n_val])
    test_keys = set(unique_keys[n_train + n_val:])

    train_indices = []
    val_indices = []
    test_indices = []

    for key_id, indices in samples_by_key.items():
        if key_id in train_keys:
            train_indices.extend(indices)
        elif key_id in val_keys:
            val_indices.extend(indices)
        elif key_id in test_keys:
            test_indices.extend(indices)

    return train_indices, val_indices, test_indices


def evaluate_model(model, dataloader, device):
    """
    Evaluate model and return predictions and true labels
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, labels in dataloader:
            X = X.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            Y_pred = model(X)
            pred_y = Y_pred.argmax(dim=1)

            all_preds.extend(pred_y.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)

    return all_preds, all_labels, accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, accuracy, title, save_path):
    """
    Plot and save confusion matrix as PDF
    """
    cm = confusion_matrix(y_true, y_pred)


    # Create figure
    plt.figure(figsize=(20, 20))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,  # Show numbers
        fmt='d',  # Integer format
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )

    # Add title with accuracy
    plt.title(f'{title}\nAccuracy: {accuracy:.2f}%', fontsize=14, pad=20, weight='bold')
    plt.ylabel('True Label', fontsize=12, weight='bold')
    plt.xlabel('Predicted Label', fontsize=12, weight='bold')

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    log.info(f"✓ Saved: {save_path}")

    return cm


def main():
    cfg = get_configuration()

    dataset_path = abspath(cfg.get('DATASET_PATH', '_dataset'))
    device = get_current_device()
    batch_size = int(cfg.get('BATCH_SIZE', 64))
    d_model = int(cfg.get('D_MODEL', 16))
    hidden = int(cfg.get('HIDDEN', 32))
    num_layers = int(cfg.get('NUM_LAYERS', 2))
    dropout = float(cfg.get('DROPOUT', 0.2))

    log.info(f'Using device: {device}')

    # Load dataset
    log.info(f"Loading dataset from: {dataset_path}")
    full_dataset = Dataset(root=dataset_path)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    log.info(f"Classes: {class_names}")

    # Create dirs
    if not os.path.isdir("results"):
        os.makedirs("results")

    # ==================== SCENARIO A ====================
    log.info("\n" + "=" * 60)
    log.info("SCENARIO A: Key-Reuse with Random Split")
    log.info("=" * 60)

    # Use same random split as in training
    split_ratio = float(cfg.get('TRAIN_VAL_SPLIT', 0.8))
    train_ds_a, val_ds_a = random_split(
        full_dataset,
        [split_ratio, 1.0 - split_ratio],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader_a = DataLoader(
        val_ds_a,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # Load Scenario A model
    model_a = CipherClassifier(
        num_classes=num_classes,
        d_model=d_model,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model_a.load_state_dict(torch.load('model_scenario_a.pth', map_location=device))
    log.info("Loaded model.pth for Scenario A")

    # Evaluate
    preds_a, labels_a, acc_a = evaluate_model(model_a, val_loader_a, device)
    log.info(f"Scenario A Accuracy: {acc_a:.2f}%")

    # Plot confusion matrix
    cm_a = plot_confusion_matrix(
        y_true=labels_a,
        y_pred=preds_a,
        class_names=class_names,
        accuracy=acc_a,
        title='Scenario A: Key-Reuse with Random Split',
        save_path='results/confusion_scenario_a.pdf'
    )

    # ==================== SCENARIO B ====================
    log.info("\n" + "=" * 60)
    log.info("SCENARIO B: Key-Disjoint Split")
    log.info("=" * 60)

    # Key-disjoint split
    train_idx, val_idx, test_idx = create_key_disjoint_split(full_dataset)

    val_ds_b = Subset(full_dataset, val_idx)

    val_loader_b = DataLoader(
        val_ds_b,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # Load Scenario B model
    model_b = CipherClassifier(
        num_classes=num_classes,
        d_model=d_model,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model_b.load_state_dict(torch.load('model_scenario_b.pth', map_location=device))
    log.info("Loaded model_scenario_b.pth for Scenario B")

    # Evaluate
    preds_b, labels_b, acc_b = evaluate_model(model_b, val_loader_b, device)
    log.info(f"Scenario B Accuracy: {acc_b:.2f}%")

    # Plot confusion matrix
    cm_b = plot_confusion_matrix(
        y_true=labels_b,
        y_pred=preds_b,
        class_names=class_names,
        accuracy=acc_b,
        title='Scenario B: Key-Disjoint Split',
        save_path='results/confusion_scenario_b.pdf'
    )

    # ==================== SUMMARY ====================
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Scenario A (Key-Reuse):      {acc_a:.2f}%")
    log.info(f"Scenario B (Key-Disjoint):   {acc_b:.2f}%")
    log.info(f"Difference:                  {acc_a - acc_b:.2f}%")
    log.info("\nGenerated files:")
    log.info("  ✓ confusion_scenario_a.pdf")
    log.info("  ✓ confusion_scenario_b.pdf")
    log.info("=" * 60)

    # Print confusion matrices
    print("\nConfusion Matrix A:")
    print(cm_a)
    print("\nConfusion Matrix B:")
    print(cm_b)


if __name__ == '__main__':
    main()