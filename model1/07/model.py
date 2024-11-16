import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, eos):
        super(Model, self).__init__()
        self.eos_token = eos
        self.hidden_size = hidden_size
        self.lstmcell = nn.LSTMCell(input_size+7, hidden_size)
        self.relu = nn.ReLU()
        self.lstmcell2 = nn.LSTMCell(hidden_size, hidden_size)

        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, signal, input, initstate):

        signal = signal.squeeze(0)

        if initstate == None:
            hx, cx = torch.zeros(signal.size(0), self.hidden_size), torch.zeros(signal.size(0), self.hidden_size)
        else:
            hx, cx = initstate

        input = input.expand(signal.size(0), -1)

        # print(input.size())
        # print(input)
        # print(signal.size())
        # print(signal)
        input_combined = torch.cat((signal, input), 1)
        # print(input_combined.size())

        hx, cx = self.lstmcell(input_combined, (hx, cx))
        x = self.relu(hx)
        hx, cx = self.lstmcell2(x, (hx, cx))
        output = self.fc(hx)
        output = self.softmax(output)


        # x = torch.argmax(output, dim=1).float()

        # output_sequence = torch.stack(output_sequence, dim=0)

        return output, (hx, cx)
