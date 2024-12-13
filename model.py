from torch import nn

class Model(nn.Module):
    INPUT_SIZE = 256

    def __init__(self, classes):
        super(Model, self).__init__()
        self.rnn = nn.GRU(self.INPUT_SIZE, 128, batch_first=True)
        self.linear = nn.Linear(128, 100)
        self.tanh = nn.ReLU()
        #self.pool = nn.MaxPool1d(64)
        self.linear2 = nn.Linear(100, classes)
        self.softmax = nn.Softmax()

    def forward(self, batch):
        batch, _ = self.rnn(batch)
        batch = self.linear(batch)
        batch = self.tanh(batch)
        batch = self.linear2(batch)
        batch = self.softmax(batch)
        return batch
