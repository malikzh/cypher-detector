#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset
from model import CipherClassifier
from config import get_configuration
from  loguru import logger as log
from os.path import abspath
import numpy as np

def get_current_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    np.random.seed(42)
    writer = SummaryWriter(log_dir='runs/scenario_a')
    cfg = get_configuration()

    # Load configurations
    dataset_path = abspath(cfg.get('DATASET_PATH', '_dataset'))
    device = get_current_device()
    split_ratio = float(cfg.get('TRAIN_VAL_SPLIT', 0.8))
    epochs = int(cfg.get('EPOCHS', 10))
    batch_size = int(cfg.get('BATCH_SIZE', 64))
    d_model = int(cfg.get('D_MODEL', 16))
    hidden = int(cfg.get('HIDDEN', 32))
    num_layers = int(cfg.get('NUM_LAYERS', 2))
    dropout = float(cfg.get('DROPOUT', 0.2))
    ckpt_path = cfg.get('CKPT_PATH', 'model_scenario_a.pth')

    log.info('Using device: {}'.format(device))

    # Подготовка датасета
    log.info("Loading dataset from: {}".format(dataset_path))
    full_dataset = Dataset(root=dataset_path)
    train_ds, val_ds = random_split(full_dataset, [split_ratio, 1.0 - split_ratio],
        generator=torch.Generator().manual_seed(42))


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory= (device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory= (device.type == 'cuda'),
    )

    log.info("Dataset loaded. Total samples: {}".format(len(full_dataset)))
    log.info("Train samples: {}, Validation samples: {}".format(len(train_ds), len(val_ds)))

    # Model configuration
    num_classes = int(len(full_dataset.classes))

    log.info("Initializing model with parameters: num_classes={}, d_model={}, hidden={}, num_layers={}, dropout={}"
             .format(num_classes, d_model, hidden, num_layers, dropout))

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

    log.info("Starting training for {} epochs".format(epochs))
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        log.info("Epoch {}/{}".format(epoch, epochs))

        # Training loop
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

        log.info("Train Loss: {:.4f} | Train Acc: {:.2f}%".format(mean_train_loss, train_acc))

        # Validation loop
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
        log.info("Val   Loss: {:.4f} | Val   Acc: {:.2f}%".format(mean_val_loss, val_acc))

        # Save best model
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), ckpt_path)
            log.info(f'Saved best model to: {ckpt_path}')

        # Save metrics
        writer.add_scalar("Loss/train", mean_train_loss, epoch)
        writer.add_scalar("Loss/val", mean_val_loss, epoch)
        writer.add_scalar("Accuracy/train %", train_acc, epoch)
        writer.add_scalar("Accuracy/val %", val_acc, epoch)
        writer.flush()


if __name__ == '__main__':
    main()
