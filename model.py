import torch.nn as nn

class CipherClassifier(nn.Module):
    def __init__(self, num_classes=4, d_model=64, hidden=128, num_layers=2, bidir=False, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)  # вход — байт 0..255
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, padded, lengths):
        # padded: [B, T], lengths: [B]
        x = self.embed(padded)  # [B, T, d_model]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.gru(packed)
        # h_n: [num_layers * num_dirs, B, hidden]
        h_last = h_n[-1] if self.gru.bidirectional is False else torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.head(h_last)  # [B, num_classes] — логиты, без softmax
