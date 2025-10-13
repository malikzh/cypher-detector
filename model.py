import torch
from torch import nn

class CipherClassifier(nn.Module):
    def __init__(self, num_classes=4, d_model=32, hidden=96, num_layers=1, bidir=False, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidir, dropout=dropout,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, padded, lengths=None):
        # padded: [B, T] (long, 0..255)
        x = self.embed(padded).contiguous()       # [B, T, d_model]; contiguous для MPS
        out, h_n = self.gru(x)                    # без pack — на MPS стабильнее
        if self.gru.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        return self.head(h_last)                  # логиты