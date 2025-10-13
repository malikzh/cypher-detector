from torch import nn
from config import get_configuration

CFG = get_configuration()

class CypherDetectorTransformerModel(nn.Module):
    def __init__(self, classes):
        super(CypherDetectorTransformerModel, self).__init__()

        seq_len = CFG['SEQUENCE_LENGTH']
        hidden_size = 32
        nhead = 10
        num_layers = 2

        encoder_layer = nn.TransformerEncoderLayer(d_model=20, nhead=nhead, dim_feedforward=hidden_size, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(20 * seq_len, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        batch = self.transformer_encoder(batch)
        batch = self.flatten(batch)
        batch = self.linear1(batch)
        batch = self.softmax(batch)
        return batch

class CypherDetectorRNNModel(nn.Module):
    def __init__(self, classes):
        super(CypherDetectorRNNModel, self).__init__()

        seq_len = CFG['SEQUENCE_LENGTH']
        hidden_size = 32

        self.gru = nn.GRU(20, hidden_size, batch_first=True, num_layers=2, dropout=0.3, bidirectional=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(hidden_size * seq_len * 2, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        batch, _ = self.gru(batch)
        batch = self.flatten(batch)
        batch = self.linear1(batch)
        batch = self.softmax(batch)
        return batch


class CypherDetectorSimpleModel(nn.Module):
    def __init__(self, classes):
        super(CypherDetectorSimpleModel, self).__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        batch = self.flatten(batch)
        batch = self.linear1(batch)
        batch = self.softmax(batch)
        return batch