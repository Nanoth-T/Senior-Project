import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

#---------------TransformerModel------------------------------#

class TransformerConv2dModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_dict, max_length,
                 encoder_heads, encoder_layers, decoder_heads, decoder_layers):
        super(TransformerConv2dModel, self).__init__()
        
        self.encoder = TransformerEncoderConv2d(input_size, hidden_size, encoder_heads, encoder_layers)
        self.decoder = TransformerDecoder(hidden_size, output_dict, max_length, decoder_heads, decoder_layers)
        
    def forward(self, signal, target=None, max_loop=None):
        encoder_output = self.encoder(signal)
        decoder_output = self.decoder(encoder_output, target, max_loop)
        return decoder_output
    
class TransformerLinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_dict, max_length,
                 encoder_heads, encoder_layers, decoder_heads, decoder_layers):
        super(TransformerLinearModel, self).__init__()
        
        self.encoder = TransformerEncoderLinear(input_size, hidden_size, encoder_heads, encoder_layers)
        self.decoder = TransformerDecoder(hidden_size, output_dict, max_length, decoder_heads, decoder_layers)
        
    def forward(self, signal, target=None, max_loop=None):
        encoder_output = self.encoder(signal)
        decoder_output = self.decoder(encoder_output, target, max_loop)
        return decoder_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * torch.sqrt(torch.tensor(self.d_model))
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x
    

class TransformerEncoderConv2d(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_heads, encoder_layers):
        super(TransformerEncoderConv2d, self).__init__()
        self.d_model = hidden_size #size after embeding and pe
        self.nhead = encoder_heads 
        self.num_layers = encoder_layers
        self.hidden_size = hidden_size
        self.n_mels = 256

        self.input_proj = nn.Conv2d(1, hidden_size, kernel_size=(self.n_mels, 1))
        self.position_encoding = PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.nhead, batch_first=True),
            num_layers=self.num_layers)

    def forward(self, signal):
        # Remove the singleton dimension
        signal = signal.squeeze(2)  # [batch, seq_len, n_mels, time_length]
        
        # Reshape for Conv2d (batch*seq_len, 1, n_mels, time_length)
        batch_size, seq_len, n_mels, time_length = signal.shape
        signal = signal.view(batch_size * seq_len, 1, n_mels, time_length)

        # Apply Conv2d
        signal_proj = self.input_proj(signal)  # Output: [batch*seq_len, hidden_size, 1, time_length]
        signal_proj = signal_proj.squeeze(2)  # Remove the second dimension: [batch*seq_len, hidden_size, time_length]

        # Reshape back to [batch, seq_len, time_length, hidden_size]
        signal_proj = signal_proj.permute(0, 2, 1).view(batch_size, seq_len, time_length, self.hidden_size)
        signal_proj = signal_proj.mean(dim=2)  # [batch, seq_len, hidden_size]

        # Apply positional encoding and transformer encoder
        signal_pe = self.position_encoding(signal_proj)
        encoder_output = self.encoder_layer(signal_pe)

        return encoder_output

class TransformerEncoderLinear(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_heads, encoder_layers):
        super(TransformerEncoderLinear, self).__init__()
        self.d_model = hidden_size #size after embeding and pe
        self.nhead = encoder_heads 
        self.num_layers = encoder_layers
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.position_encoding = PositionalEncoding(self.d_model)
        # self.dropout = nn.Dropout(p=0.3)
        self.encoder_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.nhead, batch_first=True),
            num_layers=self.num_layers)

    def forward(self, signal):
        signal_proj = self.input_proj(signal)
        signal_pe = self.position_encoding(signal_proj)
        # signal_pe = self.dropout(signal_pe)
        encoder_output = self.encoder_layer(signal_pe)

        return encoder_output

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, output_dict, max_length, decoder_heads, decoder_layers):
        super(TransformerDecoder, self).__init__()

        self.all_note = output_dict["all_note"]
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = output_dict["n_note"]
        self.nhead = decoder_heads
        self.num_layers = decoder_layers
        self.d_model = hidden_size

        self.embedding = nn.Embedding(self.output_size, self.d_model)
        self.position_encoding = PositionalEncoding(self.d_model, max_length)
        # self.dropout = nn.Dropout(p=0.3)
        self.decoder_layer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=self.nhead, batch_first=True),
            num_layers=self.num_layers)
        self.fc_out = nn.Linear(self.d_model, self.output_size)


    def forward(self, encoder_output, tgt=None, max_loop=None, tgt_mask=None):

        if tgt is not None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            tgt = self.embedding(tgt)  # Embedding layer
            tgt = self.position_encoding(tgt)  # Positional encoding
            # tgt = self.dropout(tgt)
            output = self.decoder_layer(tgt, encoder_output, tgt_mask=tgt_mask)  # Transformer decoder
            output = self.fc_out(output)  # Final linear layer to get logits for each token in the vocabulary
        else:
            # Create SOS token:
            batch_size = encoder_output.size(0)
            generated_sequence = torch.full((batch_size, 1), self.all_note.index("<SOS>"), dtype=torch.long, device=encoder_output.device)
            # decoder_input = torch.LongTensor(self.output_dict["<SOS>"]).unsqueeze(0).to(encoder_output.device)
            for _ in range(max_loop):
                tgt_mask = self.generate_square_subsequent_mask(generated_sequence.size(1)).to(encoder_output.device)
                # output = self.forward(generated_sequence, encoder_output, tgt_mask=tgt_mask)
                tokens = self.embedding(generated_sequence)
                tokens = self.position_encoding(tokens)
                # tokens = self.dropout(tokens)
                output = self.decoder_layer(tokens, encoder_output, tgt_mask=tgt_mask)
                output = self.fc_out(output) 
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

                if torch.all(next_token == self.all_note.index("<EOS>")):
                    break
        # print(output)
        return output
    
    def generate_square_subsequent_mask(self, size, device="cuda"):
        mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
#-----------------------------------------------------------#
#-----------------------------------------------------------#
#-----------------------------------------------------------#


#------------------------LSTM Model---------------------------#


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_dict, max_length):
        super(LSTMModel, self).__init__()
        
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_dict, max_length)
        
    def forward(self, signal, target=None, max_loop=None):
        encoder_output, encoder_hidden = self.encoder(signal)
        h, c = encoder_hidden
        # print(h.shape, c.shape, encoder_hidden)
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

#-----------------------------------------------------------#