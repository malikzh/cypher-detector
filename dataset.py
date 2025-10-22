import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from encoder import ENCODER_FACTORY
import numpy as np


class Dataset(Dataset):
    def __init__(self, root="_dataset"):
        self.samples = []

        # Generate class indices
        self.classes = list([class_name for class_name in ENCODER_FACTORY.keys()])

        # Traverse dataset directory
        for cipher_name in ENCODER_FACTORY.keys():
            class_path = join(root, cipher_name)
            for filename in listdir(class_path):
                if not filename.endswith(".bin"):
                    continue
                fp = join(class_path, filename)
                with open(fp, "rb") as f:
                    data = f.read()
                # Raw ciphertext in bytes: 0..255
                arr = np.frombuffer(data, dtype=np.uint8)
                seq = torch.from_numpy(arr.astype(np.int64))
                self.samples.append((cipher_name, seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_id = self.classes.index(self.samples[idx][0])
        return self.samples[idx][1], torch.tensor(class_id, dtype=torch.long)
