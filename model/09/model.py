import torch
import torch.nn as nn
import torch.nn.functional as F

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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, signal):

        signal = torch.flatten(signal, start_dim=1, end_dim=-1)

        output, hidden = self.lstm(signal)

        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(DecoderRNN, self).__init__()

        self.max_length = max_length
        self.output_size = output_size

        self.lstm = nn.LSTM(7, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        

    def forward(self, encoder_hidden, target_onehot=None):
        decoder_input = torch.zeros(1, self.output_size)
        decoder_input[0][self.output_size - 1] = 1
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_length):
            # print(decoder_input)
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            
            # print(decoder_output)

            decoder_outputs.append(decoder_output)

            use_teacher_forcing = True if torch.rand(1).item() < 0.5 else False

            if use_teacher_forcing and target_onehot is not None:
                if i >= target_onehot.size(0)-1:
                    break
                decoder_input = target_onehot[i+1] # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                # print(decoder_output)
                _, topi = decoder_output.topk(1)
                decoder_input = torch.zeros(1, self.output_size)
                decoder_input[0][topi] = 1

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        # print(decoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = F.relu(input)

        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden