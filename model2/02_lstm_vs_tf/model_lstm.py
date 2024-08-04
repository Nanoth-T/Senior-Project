import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_dict, max_length):
        super(EncoderDecoder, self).__init__()
        
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_dict, max_length)
        
    def forward(self, signal, target=None, max_loop=None):
        encoder_output, encoder_hidden = self.encoder(signal)
        h, c = encoder_hidden
        print(h.shape, c.shape, encoder_hidden)
        decoder_output, _ = self.decoder(encoder_hidden, target, max_loop)
        return decoder_output


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, signal):

        output, hidden = self.lstm(signal)

        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_dict, max_length):
        super(DecoderRNN, self).__init__()

        self.output_dict = output_dict
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = len(output_dict)

        self.lstm = nn.LSTM(self.output_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, self.output_size)


    def forward(self, encoder_hidden, target_onehot=None, max_loop=None):

        decoder_hidden = encoder_hidden
        decoder_outputs = []

        if target_onehot is not None:
            decoder_input = F.one_hot(target_onehot, num_classes=self.output_size).float()
            # print(decoder_input)
            decoder_input = decoder_input[:, :1, :]
            # print(decoder_input)
            max_loop = target_onehot.size(1)

        else:
            # Create SOS token:
            decoder_input = torch.zeros(encoder_hidden[0].size(1), 1, self.output_size)
            # print(decoder_input, decoder_input.shape)
            decoder_input[:, :, self.output_dict["<SOS>"]] = 1

        if max_loop is None:
            max_loop = self.max_length
            
        for i in range(1, max_loop+1):
            # print(i)
            # print(decoder_input)
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            # print(decoder_output)

            decoder_outputs.append(decoder_output)

            decoder_input = torch.zeros(decoder_input.size(0), 1, self.output_size)
            for b in range(decoder_input.size(0)):
                each_decoder_output = decoder_output[b]
                # print(each_decoder_output)
                _, topi = each_decoder_output.topk(1)
                if topi == self.output_dict["<EOS>"] and target_onehot is None:
                    # print("eos")
                    break
                else:
                    decoder_input[b][0][topi] = 1
            else:
                continue
            break   
            

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        # print(decoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = F.relu(input).to("cuda")
        # print(output)
        output, hidden = self.lstm(output, hidden)
        hx, cx = hidden
        # print(output)
        # print(output.size())
        output = self.out(output.view(-1, self.hidden_size))
        output_tensor = output.view(input.size(0), 1, self.output_size)
        # print(f"out:{output_tensor}")
        return output_tensor, hidden
    