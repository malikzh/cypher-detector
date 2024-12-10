from torch import nn

class Model(nn.Module):
    INPUT_SIZE = 128
    HIDDEN_SIZE = 100

    def __init__(self):
        self.rnn = nn.RNN(self.INPUT_SIZE, self.HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(self.HIDDEN_SIZE, 80)
        self.relu = nn.Tanh()
        self.pool = nn.MaxPool1d(64)
        self.linear2 = nn.Linear(64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        pass
