import torch
import torch.nn as nn
import torch.nn.functional as F

### go edit add attention decoder

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, signal):

        output, hidden = self.lstm(signal)

        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = output_size

        self.attention = BahdanauAttention(hidden_size)

        self.lstm = nn.LSTM(hidden_size+output_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout_p)



    def forward(self, encoder_outputs, encoder_hidden, target_onehot=None, max_loop=None):

        # # Create SOS token:
        # decoder_input = torch.zeros(batch_size, 1, self.output_size)
        # for i in range(batch_size):
        #     decoder_input[i][0][self.output_size - 1] = 1
        # # print(decoder_input.size())
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        if target_onehot is not None:
            decoder_input = target_onehot[:, 0]
            # print(decoder_input)
            max_loop = target_onehot.size(1)
            # print(target_onehot.size())
            # print(max_loop)
            # print(decoder_input)
        else:
            # Create SOS token:
            decoder_input = torch.zeros(encoder_hidden[0].size(1), 1, self.output_size)
            # print(decoder_input, decoder_input.shape)
            decoder_input[:, :, -1] = 1

            
        for i in range(1, max_loop+1):
            # print(i)
            # print(decoder_input)
            decoder_output, decoder_hidden, attn_weights  = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            
            # print(decoder_output)

            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            use_teacher_forcing = True if torch.rand(1).item() < 0 else False

            # for j in range(target_onehot.size(0)):
            #     print(target_onehot[j,:])

            # if use_teacher_forcing and target_onehot is not None:
            #     print("t")
            #     decoder_input = target_onehot[:, i]
                # print(decoder_input)
            # else:
                # print("no t")
            decoder_input = torch.zeros(decoder_input.size(0), 1, self.output_size)
            for b in range(decoder_input.size(0)):
                each_decoder_output = decoder_output[b]
                _, topi = each_decoder_output.topk(1)
                if topi == 5 and target_onehot is None:
                    break
                if use_teacher_forcing and target_onehot is not None:
                    decoder_input[b][0] = target_onehot[b][i]
                else:
                    decoder_input[b][0][topi] = 1
            else:
                continue
            break   
            # decoder_input = torch.zeros(decoder_input.size(0), 1, self.output_size)
            # for b in range(decoder_input.size(0)):
            #     each_decoder_output = decoder_output[b]
            #     _, topi = each_decoder_output.topk(1)
            #     if topi == 5 and target_onehot is None:
            #         break
            #     decoder_input[b][0][topi] = 1
            # else:
            #     continue
            # break
        
        # คือไรฟ่ะ
        # decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
        # decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        # print(decoder_outputs)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        input_drop = self.dropout(input)
        # print(output)
        # print(output.size())
        query = hidden[0].permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_lstm = torch.cat((input_drop, context), dim=2)

        output, hidden = self.lstm(input_lstm, hidden)
        hx, cx = hidden
        # print(output)
        # print(output.size())
        # print(hx.size(), cx.size())
        output = self.out(output.view(-1, self.hidden_size))
        # print(output.size())
        output_tensor = output.view(input.size(0), 1, self.output_size)
        # print(f"out:{output_tensor}")
        return output_tensor, hidden, attn_weights