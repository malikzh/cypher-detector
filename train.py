#!/usr/bin/env python3

from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from model import CypherDetectorRNNModel, CypherDetectorSimpleModel
from config import get_configuration
import torch

from torch.utils.tensorboard import SummaryWriter

# Initialize
writer = SummaryWriter()
cfg = get_configuration()

torch.set_default_device(cfg['DEVICE'])
print(f'Selected device: {cfg["DEVICE"]}')

# Prepare data
dataset = Dataset()

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=torch.Generator(cfg['DEVICE']))

train_loader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], generator=torch.Generator(cfg['DEVICE']), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg['BATCH_SIZE'], generator=torch.Generator(cfg['DEVICE']), shuffle=True)


# prepare model
model = CypherDetectorRNNModel(len(dataset.CLASSES))
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LEARNING_RATE'], weight_decay=cfg['WEIGHT_DECAY'])
loss_fn = torch.nn.CrossEntropyLoss()
best_loss = float('inf')

for epoch in range(cfg['EPOCHS']):
    print('Epoch {}/{}'.format(epoch+1, cfg['EPOCHS']))

    model.train(True)

    train_loss = 0.0
    print('Training...')
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
    print('Validating...')
    with torch.no_grad():
        for i, (X, Y) in enumerate(val_loader):
            predicted_Y = model(X)
            loss = loss_fn(predicted_Y, Y)
            val_loss += loss.item()
            correct += (torch.argmax(predicted_Y, dim=1) == torch.argmax(Y, dim=1)).float().sum()
            total += Y.size(0)

    mean_train_loss = train_loss / len(train_loader)
    mean_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    writer.add_scalar("Loss/train", mean_train_loss, epoch)
    writer.add_scalar("Loss/validation", mean_val_loss, epoch)
    writer.add_scalar("Accuracy/validation %", accuracy, epoch)
    print('Train Loss: {:.4f}, Val Loss: {:.4f}'.format(mean_train_loss, mean_val_loss))
    print('Val Accuracy: {:.2f}%'.format(accuracy))
    
    if val_loss < best_loss:
        print('Saving model...')
        torch.save(model.state_dict(), 'model.pth')

    writer.flush()
    