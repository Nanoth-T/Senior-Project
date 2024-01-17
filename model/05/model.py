import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size

        self.lstm_cell1 = nn.LSTMCell(input_size+7, hidden_size)
        self.relu = nn.ReLU()

        self.lstm_cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, signal, input, initstate):
        if initstate == None:
            hx, cx = torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)
        else:
            hx, cx = initstate
        signal = torch.flatten(signal)
        signal = signal.unsqueeze(0)

        input_combined = torch.cat((signal, input), 1)
        hx, cx = self.lstm_cell1(input_combined, (hx, cx))
        x = self.relu(hx)
        hx, cx = self.lstm_cell2(x, (hx, cx))
        out = self.fc(hx)
        out = self.softmax(out)

        return out, (hx, cx)
