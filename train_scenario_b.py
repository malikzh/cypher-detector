#!/usr/bin/env python3
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset
from model import CipherClassifier
from config import get_configuration
from loguru import logger as log
from os.path import abspath


def get_current_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_key_disjoint_split(dataset, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset by key_id to ensure no key overlap between train/val/test
    """
    # Group samples by key_id
    samples_by_key = {}
    for idx in range(len(dataset)):
        key_id = dataset.get_key_id(idx)
        if key_id not in samples_by_key:
            samples_by_key[key_id] = []
        samples_by_key[key_id].append(idx)

    # Get unique keys
    unique_keys = sorted(samples_by_key.keys())
    n_keys = len(unique_keys)

    log.info(f"Total unique keys: {n_keys}")
    log.info(f"Samples per key: ~{len(dataset) // n_keys}")

    # Shuffle keys
    np.random.seed(42)
    np.random.shuffle(unique_keys)

    # Split keys into train/val/test
    n_train = int(n_keys * train_ratio)
    n_val = int(n_keys * val_ratio)

    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train:n_train + n_val])
    test_keys = set(unique_keys[n_train + n_val:])

    log.info(f"Train keys: {len(train_keys)}")
    log.info(f"Val keys: {len(val_keys)}")
    log.info(f"Test keys: {len(test_keys)}")

    # Create index lists
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

    log.info(f"Train samples: {len(train_indices)}")
    log.info(f"Val samples: {len(val_indices)}")
    log.info(f"Test samples: {len(test_indices)}")

    return train_indices, val_indices, test_indices


def main():
    writer = SummaryWriter(log_dir='runs/scenario_b')
    cfg = get_configuration()

    dataset_path = abspath(cfg.get('DATASET_PATH', '_dataset'))
    device = get_current_device()
    epochs = int(cfg.get('EPOCHS', 50))
    batch_size = int(cfg.get('BATCH_SIZE', 64))
    d_model = int(cfg.get('D_MODEL', 16))
    hidden = int(cfg.get('HIDDEN', 32))
    num_layers = int(cfg.get('NUM_LAYERS', 2))
    dropout = float(cfg.get('DROPOUT', 0.2))
    ckpt_path = 'model_scenario_b.pth'

    log.info(f'Using device: {device}')
    log.info("=" * 60)
    log.info("SCENARIO B: Key-Disjoint Split Training")
    log.info("=" * 60)

    # Load dataset
    log.info(f"Loading dataset from: {dataset_path}")
    full_dataset = Dataset(root=dataset_path)

    # Key-disjoint split
    train_idx, val_idx, test_idx = create_key_disjoint_split(
        full_dataset,
        train_ratio=0.8,
        val_ratio=0.1
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    num_classes = len(full_dataset.classes)
    log.info(f"Number of classes: {num_classes}")

    model = CipherClassifier(
        num_classes=num_classes,
        d_model=d_model,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    lr = float(cfg.get('LEARNING_RATE', 3e-4))
    wd = float(cfg.get('WEIGHT_DECAY', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss()

    log.info(f"Starting training for {epochs} epochs")
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        log.info(f"Epoch {epoch}/{epochs}")

        # Training
        model.train()
        train_loss_sum = 0.0
        total_train = 0
        correct_train = 0

        for X, labels in train_loader:
            X = X.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            Y_pred = model(X)
            loss = loss_fn(Y_pred, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            total_train += len(labels)
            pred_y = Y_pred.argmax(dim=1)
            correct_train += (pred_y == labels).sum().item()

        mean_train_loss = train_loss_sum / max(1, total_train)
        train_acc = 100.0 * correct_train / max(1, total_train)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for X, labels in val_loader:
                X = X.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                Y_pred = model(X)
                loss = loss_fn(Y_pred, labels)

                val_loss_sum += loss.item() * len(labels)
                total_val += len(labels)
                pred_y = Y_pred.argmax(dim=1)
                correct_val += (pred_y == labels).sum().item()

        mean_val_loss = val_loss_sum / max(1, total_val)
        val_acc = 100.0 * correct_val / max(1, total_val)

        log.info(f"Train Loss: {mean_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        log.info(f"Val   Loss: {mean_val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            log.info(f'Saved best model to: {ckpt_path} (Val Acc: {val_acc:.2f}%)')

        # Tensorboard
        writer.add_scalar("Loss/train", mean_train_loss, epoch)
        writer.add_scalar("Loss/val", mean_val_loss, epoch)
        writer.add_scalar("Accuracy/train %", train_acc, epoch)
        writer.add_scalar("Accuracy/val %", val_acc, epoch)
        writer.flush()

    log.info("=" * 60)
    log.info(f"Training completed. Best Val Acc: {best_val_acc:.2f}%")
    log.info("=" * 60)


if __name__ == '__main__':
    main()