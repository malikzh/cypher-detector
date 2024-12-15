#!/usr/bin/env python3

from model import CypherDetectorRNNModel, CypherDetectorSimpleModel
from dataset import Dataset
from config import get_configuration
import torch
from torch.utils.data import DataLoader

# Initialize
cfg = get_configuration()

torch.set_default_device(cfg['DEVICE'])
print(f'Selected device: {cfg["DEVICE"]}')

FILE = 'Blowfish.txt' # Change file for testing

model = CypherDetectorRNNModel(3) # IMPORTANT: 3 classes

model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

dataset = Dataset(FILE)
classes_names = dataset.CLASSES_NAMES
dataloader = DataLoader(dataset, batch_size=cfg['BATCH_SIZE'], generator=torch.Generator(cfg['DEVICE']))

correct = 0
total = 0
with torch.no_grad():
    for i, (X, Y) in enumerate(dataloader):
        predicted_Y = model(X)
        predicted_class = torch.argmax(predicted_Y, dim=1)
        real_class = torch.argmax(Y, dim=1)

        correct += (predicted_class == real_class).float().sum()
        total += Y.size(0)

print(f'Total accuracy: {correct / total * 100.0:.2f}% Correct items: {int(correct)} / Total items: {int(total)}')