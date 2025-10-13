#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset, collate_varlen
from model import CipherClassifier
from config import get_configuration


def main():
    # --- init ---
    writer = SummaryWriter()
    cfg = get_configuration()

    device = torch.device(cfg.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Selected device: {device}')

    seed = int(cfg.get('SEED', 42))
    torch.manual_seed(seed)

    # --- data ---
    full_ds = Dataset(root=cfg.get('DATASET_PATH', '_dataset'))

    # split 80/20 (детерминированно)
    gen = torch.Generator().manual_seed(seed)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    # DataLoader’ы с collate_fn
    batch_size = int(cfg.get('BATCH_SIZE', 64))
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_varlen,
        num_workers=0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_varlen,
        num_workers=0,
        pin_memory=use_cuda,
    )

    # --- model ---
    num_classes = int(cfg.get('NUM_CLASSES', 4))
    d_model = int(cfg.get('D_MODEL', 64))
    hidden = int(cfg.get('HIDDEN', 128))
    num_layers = int(cfg.get('RNN_LAYERS', 2))
    bidir = bool(cfg.get('BIDIRECTIONAL', False))

    model = CipherClassifier(
        num_classes=num_classes,
        d_model=d_model,
        hidden=hidden,
        num_layers=num_layers,
        bidir=bidir,
        dropout=float(cfg.get('DROPOUT', 0.2)),
    ).to(device)

    lr = float(cfg.get('LEARNING_RATE', 3e-4))
    wd = float(cfg.get('WEIGHT_DECAY', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs = int(cfg.get('EPOCHS', 10))
    ckpt_path = cfg.get('CKPT_PATH', 'model.pth')

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')

        # --- train ---
        model.train()
        train_loss_sum = 0.0
        total_train = 0
        correct_train = 0

        for padded, lengths, labels in train_loader:
            # move to device
            padded = padded.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(padded, lengths)                # [B, C], логиты (без softmax)
            loss = loss_fn(logits, labels)                 # labels: int [0..C-1]
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            total_train += labels.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()

        mean_train_loss = train_loss_sum / max(1, total_train)
        train_acc = 100.0 * correct_train / max(1, total_train)

        # --- validate ---
        model.eval()
        val_loss_sum = 0.0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for padded, lengths, labels in val_loader:
                padded = padded.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(padded, lengths)
                loss = loss_fn(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                total_val += labels.size(0)
                preds = logits.argmax(dim=1)
                correct_val += (preds == labels).sum().item()

        mean_val_loss = val_loss_sum / max(1, total_val)
        val_acc = 100.0 * correct_val / max(1, total_val)

        # --- logs ---
        writer.add_scalar("Loss/train", mean_train_loss, epoch)
        writer.add_scalar("Loss/val", mean_val_loss, epoch)
        writer.add_scalar("Accuracy/train %", train_acc, epoch)
        writer.add_scalar("Accuracy/val %", val_acc, epoch)
        writer.flush()

        print(f'Train Loss: {mean_train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val   Loss: {mean_val_loss:.4f} | Val   Acc: {val_acc:.2f}%')

        # --- save best ---
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f'Saved best model to: {ckpt_path}')


if __name__ == "__main__":
    main()
