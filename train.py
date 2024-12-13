#!/usr/bin/env python3

from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from model import Model
from config import get_configuration
import torch

# Initialize
cfg = get_configuration()

torch.set_default_device(cfg['DEVICE'])
print(f'Selected device: {cfg["DEVICE"]}')

# Prepare data
dataset = Dataset()

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=torch.Generator(cfg['DEVICE']))

train_loader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], generator=torch.Generator(cfg['DEVICE']))
val_loader = DataLoader(val_dataset, batch_size=cfg['BATCH_SIZE'], generator=torch.Generator(cfg['DEVICE']))


# prepare model
model = Model(len(dataset.LABELS))
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LEARNING_RATE'], weight_decay=cfg['WEIGHT_DECAY'])
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(cfg['EPOCHS']):
    print('Epoch {}/{}'.format(epoch+1, cfg['EPOCHS']))

    model.train(True)

    train_loss = 0.0
    for i, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()
        predicted_Y = model(X)
        loss = loss_fn(predicted_Y, Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(val_loader):
            predicted_Y = model(X)
            loss = loss_fn(predicted_Y, Y)
            val_loss += loss.item()
            correct += (predicted_Y == Y).sum().item()
            total += Y.size(0)

    print('Train Loss: {:.4f}, Val Loss: {:.4f}'.format(train_loss / len(train_loader), val_loss / len(val_loader)))
    print('Accuracy: {}%'.format(100 * correct / total))