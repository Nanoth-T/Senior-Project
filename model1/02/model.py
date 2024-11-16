import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, signal):
        h0 = torch.zeros(1, signal.size(0), self.hidden_size)
        out, _ = self.rnn(signal, h0)
        # outputs = []
        outputs = self.fc(out)
        # for i in range(output_steps):
        #     step_output = self.fc(out[:, i, :])
        #     outputs.append(step_output.unsqueeze(1))
        
        # return torch.cat(outputs, dim=1)
        return outputs
