import numpy as np
import torch
from os import listdir
from os.path import isfile, join
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    PATH = '_dataset'
    LABELS = set()
    DATA = []
    CLASSES = {
        'AES': np.array([0.0, 0.0, 1.0]),
        'Blowfish': np.array([0.0, 1.0, 0.0]),
        'DES': np.array([1.0, 0.0, 0.0]),
    }

    CLASSES_NAMES = ['DES', 'Blowfish', 'AES']

    def __init__(self, file_full_path = None):
        for filename in (listdir(self.PATH) if file_full_path is None else [file_full_path]):
            fullpath = join(self.PATH, filename)
            label = Path(filename).stem
            self.LABELS.add(label)
            if not isfile(fullpath) or not filename.endswith('.txt'):
                continue

            with open(fullpath, 'r') as f:
                for item in f.readlines():
                    if item == '':
                        continue

                    value = [x / 255.0 for x in bytes.fromhex(item)]

                    if (len(value) > 256):
                        value = value[:256]

                    assert len(value) == 256, "Size must be 256, but has: {}".format(len(value))
                    self.DATA.append([label, value])

    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, index):
        item = self.DATA[index]
        arr = np.array(item[1], dtype='float32')
        arr = np.reshape(arr, (16, 16))
        return arr, self.CLASSES[item[0]]