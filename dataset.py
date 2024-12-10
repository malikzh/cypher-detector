import torch
from os import listdir
from os.path import isfile, join
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    PATH = '_dataset'
    LABELS = set()
    DATA = []

    def __init__(self):
        for filename in listdir(self.PATH):
            fullpath = join(self.PATH, filename)
            label = Path(filename).stem
            self.LABELS.add(label)
            if not isfile(fullpath) or not filename.endswith('.txt'):
                continue

            with open(fullpath, 'r') as f:
                for item in f.readlines():
                    value = [x / 255.0 for x in bytes.fromhex(item)]
                    self.DATA.append([label, value])

    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, index):
        item = self.DATA[index]
        return item[1], item[0]