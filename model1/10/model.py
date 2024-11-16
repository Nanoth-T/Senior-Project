import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, signal):
        # print(signal.size())

        signal = torch.flatten(signal, start_dim=2, end_dim=-1)

        output, hidden = self.lstm(signal)

        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(DecoderRNN, self).__init__()

        self.max_length = max_length
        self.output_size = output_size

        self.lstm = nn.LSTM(7, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        

    def forward(self, encoder_hidden, target_onehot=None):
        decoder_input = torch.zeros(10, 1, self.output_size)
        for i in range(10):
            decoder_input[i][0][self.output_size - 1] = 1
        # print(decoder_input.size())
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_length):
            # print(decoder_input)
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            
            # print(decoder_output)

            decoder_outputs.append(decoder_output)

            use_teacher_forcing = True if torch.rand(1).item() < 0.5 else False

            decoder_input = torch.zeros(10, 1, self.output_size)
            for b in range(10):
                each_decoder_output = decoder_output[b]
                if use_teacher_forcing and target_onehot is not None:
                    if i >= target_onehot.size(0)-1:
                        break
                    decoder_input[b] = target_onehot[i+1] # Teacher forcing --> ยังไม่ได้แก้
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    # print(decoder_output)
                    _, topi = each_decoder_output.topk(1)
                    decoder_input[b][0][topi] = 1

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        # print(decoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = F.relu(input)
        # print(output)
        # print(output.size())
        output, hidden = self.lstm(output, hidden)
        hx, cx = hidden
        # print(output)
        # print(output.size())
        # print(hx.size(), cx.size())
        output = self.out(output.view(-1, 128))
        # print(output.size())
        output_tensor = output.view(10, 1, 7)
        # print(f"out:{output_tensor}")
        return output_tensor, hidden