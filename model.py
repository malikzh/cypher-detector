from torch import nn

class CypherDetectorModel(nn.Module):
    def __init__(self, classes):
        super(CypherDetectorModel, self).__init__()

        self.gru = nn.GRU(16, 8, batch_first=True)
        self.linear1 = nn.Linear(128, classes)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        batch, _ = self.gru(batch)
        batch = self.flatten(batch)
        batch = self.linear1(batch)
        batch = self.softmax(batch)
        return batch
