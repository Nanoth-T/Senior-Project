import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.o2o = nn.Linear(hidden_size + output_size, output_size)
        # self.dropout = nn.Dropout(0.1)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, signal, output_steps):
        h0 = torch.zeros(1, signal.size(0), self.hidden_size)
        # input_combined = torch.cat((signal, input, hidden), 1)
        # hidden = self.i2h(input_combined)
        # print(hidden.shape)
        # output = self.i2o(input_combined)
        # print(output.shape)
        # output_combined = torch.cat((hidden, output), 1)
        # output = self.o2o(output_combined)
        # output = self.dropout(output)
        # output = self.softmax(output)
        out, _ = self.rnn(signal, h0)
        outputs = []
        for i in range(output_steps):
            step_output = self.fc(out[:, i, :])
            outputs.append(step_output.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

    # def initHidden(self):
    #     return torch.zeros(1, self.hidden_size)
