import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_dict, max_length):
        super(EncoderDecoder, self).__init__()
        
        self.encoder = TransformerEncoder(input_size, hidden_size)
        self.decoder = TransformerDecoder(hidden_size, output_dict, max_length)
        
    def forward(self, signal, target=None, max_loop=None):
        encoder_output = self.encoder(signal)
        decoder_output = self.decoder(encoder_output, target, max_loop)
        return decoder_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
    

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransformerEncoder, self).__init__()
        self.d_model = hidden_size #size after embeding and pe
        self.nhead = 1 
        self.num_layers = 1
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.position_encoding = PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.nhead, batch_first=True),
            num_layers=self.num_layers)

    def forward(self, signal):
        signal_proj = self.input_proj(signal)
        signal_pe = self.position_encoding(signal_proj)
        encoder_output = self.encoder_layer(signal_pe)

        return encoder_output


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, output_dict, max_length):
        super(TransformerDecoder, self).__init__()

        self.output_dict = output_dict
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = len(output_dict)
        self.nhead = 1
        self.num_layers = 1
        self.d_model = hidden_size

        self.embedding = nn.Embedding(self.output_size, self.d_model)
        self.position_encoding = PositionalEncoding(self.d_model)
        self.decoder_layer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=self.nhead, batch_first=True),
            num_layers=self.num_layers)
        self.fc_out = nn.Linear(self.d_model, self.output_size)


    def forward(self, encoder_output, tgt=None, max_loop=None, tgt_mask=None):

        if tgt is not None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            tgt = self.embedding(tgt)  # Embedding layer
            tgt = self.position_encoding(tgt)  # Positional encoding
            output = self.decoder_layer(tgt, encoder_output, tgt_mask=tgt_mask)  # Transformer decoder
            output = self.fc_out(output)  # Final linear layer to get logits for each token in the vocabulary
        else:
            # Create SOS token:
            batch_size = encoder_output.size(0)
            generated_sequence = torch.full((batch_size, 1), self.output_dict["<SOS>"], dtype=torch.long, device=encoder_output.device)
            # decoder_input = torch.LongTensor(self.output_dict["<SOS>"]).unsqueeze(0).to(encoder_output.device)
            for _ in range(max_loop):
                tgt_mask = self.generate_square_subsequent_mask(generated_sequence.size(1)).to(encoder_output.device)
                # output = self.forward(generated_sequence, encoder_output, tgt_mask=tgt_mask)
                tokens = self.embedding(generated_sequence)
                tokens = self.position_encoding(tokens)
                output = self.decoder_layer(tokens, encoder_output, tgt_mask=tgt_mask)
                output = self.fc_out(output) 
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

                if torch.all(next_token == self.output_dict["<EOS>"]):
                    break
        # print(output)
        return output
    
    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    