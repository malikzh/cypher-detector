import numpy as np
import torch
from os import listdir
from os.path import isfile, join
from pathlib import Path


class Dataset(torch.utils.data.Dataset):
    PATH = '_dataset'
    ITEMS = []
    CLASSES = {
        '3DES': np.array(     [1.0, 0.0, 0.0, 0.0]),
        'AES': np.array(      [0.0, 1.0, 0.0, 0.0]),
        'Kuznechik': np.array([0.0, 0.0, 1.0, 0.0]),
        'TwoFish': np.array(  [0.0, 0.0, 0.0, 1.0]),
    }

    def __init__(self, file_full_path=None):
        for classname, Y in self.CLASSES.items():
            class_path = join(self.PATH, classname)

            for filename in listdir(class_path):
                if not filename.endswith('.bin'):
                    continue

                file_path = join(class_path, filename)
                with open(file_path, 'rb') as f:
                    arr = np.frombuffer(f.read(), dtype=np.uint8)
                    X = (arr == 255).astype(np.float32)
                    self.ITEMS.append((X, Y))

        max_len = max(len(X) for X, _ in self.ITEMS)

        padded_items = []
        for X, Y in self.ITEMS:
            if len(X) < max_len:
                padded = np.pad(X, (0, max_len - len(X)), mode="constant")
            else:
                padded = X
            padded_items.append((padded, Y))

        self.ITEMS = padded_items

    def __len__(self):
        return len(self.ITEMS)

    def __getitem__(self, index):
        item = self.ITEMS[index]
        return np.reshape(item[0], (52, 20)), item[1]
