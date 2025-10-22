import torch
from torch import nn


class CipherClassifier(nn.Module):
    def __init__(self, num_classes=4, d_model=32, hidden=96, num_layers=1, dropout=0.1, num_channels=1):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)

        self.conv1d_8 = nn.Conv1d(in_channels=num_channels, out_channels=d_model, kernel_size=8, stride=2)
        self.conv1d_16 = nn.Conv1d(in_channels=num_channels, out_channels=d_model, kernel_size=16, stride=2)

        self.transformerEncoder = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=54, nhead=6, dim_feedforward=hidden, dropout=dropout,
                                           activation='relu', batch_first=True),
                num_layers=num_layers,
            ),
            nn.Flatten()
        )

        self.output = nn.Sequential(
            nn.Linear(3456, 1500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1500, 250),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(250, num_classes)
        )

    def forward(self, x):
        x = self.embed(x).contiguous()
        x_8 = self.conv1d_8(x)
        x_16 = self.conv1d_16(x)

        x = torch.concat([x_8, x_16], dim=2)
        x = self.transformerEncoder(x)

        return self.output(x)