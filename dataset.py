import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join
import numpy as np

class Dataset(Dataset):
    def __init__(self, root="_dataset"):
        self.class2id = {
            "3DES": 0,
            "AES": 1,
            "Kuznechik": 2,
            "TwoFish": 3,
        }
        self.samples = []  # список (bytes_tensor[int64], length:int, label:int)

        for cls_name, y in self.class2id.items():
            class_path = join(root, cls_name)
            for filename in listdir(class_path):
                if not filename.endswith(".bin"):
                    continue
                fp = join(class_path, filename)
                with open(fp, "rb") as f:
                    data = f.read()
                # Сырой шифртекст в байтах: 0..255
                arr = np.frombuffer(data, dtype=np.uint8)
                seq = torch.from_numpy(arr.astype(np.int64))  # [L], dtype long для Embedding
                self.samples.append((seq, len(seq), y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, L, y = self.samples[idx]
        return seq, L, y

from torch.nn.utils.rnn import pad_sequence

def collate_varlen(batch):
    # batch: list of (seq[L_i], L_i, y_i)
    seqs, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels  = torch.tensor(labels,  dtype=torch.long)

    # паддим справа нулями (0 — норм, мы всё равно эмбеддим байты)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)  # [B, max_len]
    return padded, lengths, labels