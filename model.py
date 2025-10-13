from torch import nn

class CypherDetectorRNNModel(nn.Module):
    def __init__(self, classes):
        super(CypherDetectorRNNModel, self).__init__()

        self.gru = nn.GRU(20, 16, batch_first=True, num_layers=2, dropout=0.3, bidirectional=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(832 * 2, classes)
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