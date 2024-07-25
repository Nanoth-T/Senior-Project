import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_dict, max_length, num_layers, nhead):
        super(EncoderDecoder, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size, num_layers, nhead, max_length)
        self.decoder = TransformerDecoder(hidden_size, output_dict, num_layers, nhead, max_length)

    def forward(self, signal, tgt=None):
        encoder_output = self.encoder(signal)
        # print(signal.shape)
        # print(encoder_output.shape)
        decoder_output, _ = self.decoder(encoder_output, tgt)
        return decoder_output

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_length):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_length, embedding_size)
        pe = torch.zeros(max_length, embedding_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_length, embedding_size)
        
        # Register as buffer so that it is not a parameter but part of the state dict
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, nhead, max_length):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)  # Project input features to hidden size
        self.pos_encoder = PositionalEncoding(hidden_size, max_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead), 
            num_layers=num_layers
        )
        self.hidden_size = hidden_size

    def forward(self, signal):
        signal = self.input_proj(signal) * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        signal = self.pos_encoder(signal)  # (seq_len, batch, dim)
        output = self.transformer_encoder(signal)
        return output
    

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, output_dict, num_layers, nhead, max_length):
        super(TransformerDecoder, self).__init__()
        self.max_length = max_length
        self.output_dict = output_dict
        self.output_dict_swap = {v: k for k, v in output_dict.items()}
        self.output_size = len(output_dict)
        self.target_embedding = nn.Linear(self.output_size, hidden_size)  # Project one-hot to hidden size
        self.pos_encoder = PositionalEncoding(hidden_size, max_length)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers=num_layers
        )
        self.out = nn.Linear(hidden_size, self.output_size)  # Output layer to predict token probabilities

    def forward(self, encoder_output, tgt=None):
        if tgt is not None:
            tgt = F.one_hot(tgt, num_classes=self.output_size).float()  # Convert integers to one-hot
            # print(tgt, tgt.shape)
            tgt = self.target_embedding(tgt)  # Embed target tokens
            tgt = self.pos_encoder(tgt)  # (seq_len, batch, dim)
            tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
            output = self.transformer_decoder(tgt, encoder_output, tgt_mask)
            output = self.out(output)  # (seq_len, batch, output_size)
        else:
            # Inference mode
            current = "SOS"
            tgt_list = []
            
            for i in range(self.max_length):
                tgt_list.append(self.output_dict[current])
                tgt = F.one_hot(torch.LongTensor(tgt_list), num_classes=self.output_size).float().unsqueeze(1).to("cuda")
                # print(tgt)
                tgt = self.target_embedding(tgt)  # Embed target tokens
                tgt = self.pos_encoder(tgt)  # (seq_len, batch, dim)
                # print(tgt.shape)
                tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
                output = self.transformer_decoder(tgt, encoder_output, tgt_mask)
                output = self.out(output)  # (seq_len, batch, output_size)
                # print(output)
                topv, topi = output[-1].topk(1)
                current = self.output_dict_swap[topi.item()]
                # print(current)
                if current == "EOS":
                    break

        return output, None


    