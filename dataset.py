import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from encoder import ENCODER_FACTORY
import numpy as np
import re


class Dataset(Dataset):
    def __init__(self, root="_dataset"):
        self.samples = []

        # Generate class indices
        self.classes = list([class_name for class_name in ENCODER_FACTORY.keys()])
        self.classes = sorted(self.classes)

        # Traverse dataset directory
        for cipher_name in ENCODER_FACTORY.keys():
            class_path = join(root, cipher_name)
            for filename in listdir(class_path):
                if not filename.endswith(".bin"):
                    continue
                fp = join(class_path, filename)
                with open(fp, "rb") as f:
                    data = f.read()

                match = re.search(r'_key(\d+)\.bin', filename)
                if match:
                    key_id = int(match.group(1))
                else:
                    key_id = -1  # fallback

                file_path = join(class_path, filename)

                self.samples.append((file_path, self.classes.index(cipher_name), key_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, key_id = self.samples[idx]

        with open(file_path, 'rb') as f:
            data = f.read()

        X = torch.tensor(list(data), dtype=torch.long)
        return X, label

    def get_key_id(self, idx):
        """Return key_id for sample at index idx"""
        return self.samples[idx][2]