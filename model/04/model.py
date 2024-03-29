import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size+7, hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size+7, output_size)
        self.o2o = nn.Linear(hidden_size+output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.rnn = nn.RNN(input_size, hidden_size)
        # self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, signal, input, hidden):
        signal = torch.flatten(signal)
        signal = signal.unsqueeze(0)
        # print(hidden.shape)
        # print(signal.shape)
        # print(input.shape)
        input_combined = torch.cat((signal, input, hidden), 1)
        output = self.i2o(input_combined)
        hidden = self.i2h(input_combined)
        # print(hidden.shape)
        # print(output.shape)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
