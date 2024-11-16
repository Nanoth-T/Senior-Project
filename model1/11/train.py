import torch
import torch.nn as nn
from model import EncoderRNN, DecoderRNN
from preprocessing import MusicDataset, create_data_loader
from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
import math
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import pathlib


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def visualize_mel_spectrogram(tensor, title="Mel Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(tensor[0].numpy(), cmap='viridis', origin='lower', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel Frequency Bin')
    plt.tight_layout()
    return fig

def train(encoder, decoder, num_epochs, sound, criterion, 
          encoder_optimizer, decoder_optimizer, writer):

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        total_loss = 0
        for batch in sound:
            signal_tensor, target_onehot, target_tensor = batch

            # mel_spectrogram = visualize_mel_spectrogram(signal_tensor, f'Mel Spectrogram - Epoch {epoch+1}')
            # writer.add_figure(f'Mel Spectrogram/{epoch * len(sound)}', mel_spectrogram, global_step=None, close=True, walltime=None)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # print(target_tensor)

            encoder_outputs, encoder_hidden = encoder(signal_tensor)
            # print(encoder_hidden)
            decoder_outputs, _ = decoder(encoder_hidden, target_onehot=target_onehot)

            loss = criterion(
                    decoder_outputs.view(-1, 7),
                    target_tensor.view(-1)
                    )

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()



            total_loss += loss.item()

            # for name, param in model.named_parameters():
            #     writer.add_histogram(name + '/gradients', param.grad, epoch * len(sound))

            writer.add_scalar('Loss/train', loss.item(), epoch * len(sound))

            # for name, param in model.named_parameters():
            #     writer.add_histogram(name, param.data, epoch * len(sound))
            
            # fixed += 1
        
        print(f"{timeSince(start)} (Epoch {epoch+1}/{num_epochs}), Loss: {total_loss}")




if __name__ == "__main__":

    current_path = pathlib.Path(__file__).parent.resolve()
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    AUDIO_DIR = "dataset/02"
    ANNOTATIONS_FILE = AUDIO_DIR + "/metadata.csv"
    writer = SummaryWriter(f'{str(current_path)}/log/{str(AUDIO_DIR)}/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}')
    SAMPLE_RATE = 22050
    time_length = 200
    max_length = 60
    batch_size = 3
    n_mels = 256

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 2048,
        hop_length = 1025,
        n_mels = n_mels
        )


    sound = MusicDataset(ANNOTATIONS_FILE,
                    AUDIO_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    rhythm_dict,
                    max_length
                    )

    # print(f"There are {len(sound)} samples in the dataset.")
    # for i in range(len(sound)):
    #     signal,target_onehot, label = sound[i]
    #     print(signal.shape)

    train_data_loader = create_data_loader(sound, batch_size)

    # print(data_loader)

    input_size = n_mels*time_length
    hidden_size = 128
    output_size = len(rhythm_dict)
    max_length = 60


    encoder = EncoderRNN(input_size, hidden_size)
    decoder = DecoderRNN(hidden_size, output_size, max_length)

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
    num_epochs = 100
    start = time.time()


    train(encoder, decoder, num_epochs, train_data_loader, criterion, encoder_optimizer, decoder_optimizer, writer)
    torch.save(encoder.state_dict(), f"{str(current_path)}/enmodel.pth")
    torch.save(decoder.state_dict(), f"{str(current_path)}/demodel.pth")
    print("Model trained and stored at model.pth")

    
