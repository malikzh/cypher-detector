import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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

MAX_T = 4096

def collate_varlen(batch):
    seqs, lengths, labels = zip(*batch)

    fixed = []
    fixed_L = []
    for s, L in zip(seqs, lengths):
        # защита от пустых
        if L is None or L <= 0 or s.numel() == 0:
            s = torch.zeros(1, dtype=torch.long); L = 1

        # cap/окно
        if L > MAX_T:
            start = np.random.randint(0, L - MAX_T + 1)
            s = s[start:start+MAX_T]
            L = MAX_T
        else:
            # на всякий — вдруг за пределами
            s = s[:L]

        fixed.append(s)
        fixed_L.append(int(min(L, s.numel())))

    lengths_t = torch.tensor(fixed_L, dtype=torch.long)
    padded = pad_sequence(fixed, batch_first=True, padding_value=0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded, lengths_t, labels_t