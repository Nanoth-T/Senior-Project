import torch
import torch.nn as nn
from model import RNN
from preprocessing import MusicDataset
from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
import math
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


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

def train(model, num_epochs, sound, criterion, optimizer, writer):
    # Training loop
    signal_tensor, input_tensor, target_tensor = sound
    target_tensor.unsqueeze_(-1)
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        # for signal_tensor, input_tensor, target_tensor in sound:
        #     target_tensor.unsqueeze_(-1)
        for i in range(1):
            
            # print(signal_tensor.shape, input_tensor.shape)
            mel_spectrogram = visualize_mel_spectrogram(signal_tensor, f'Mel Spectrogram - Epoch {epoch+1}')
            writer.add_figure(f'Mel Spectrogram/{epoch * len(sound)}', mel_spectrogram, global_step=None, close=True, walltime=None)

            # if fixed == 1:
            #     break
            
            hidden = model.initHidden()
            # print(hidden)
            optimizer.zero_grad()
            loss = torch.Tensor([0])
            # print(target_tensor)
            for i in range(input_tensor.size(0)):
                # print(input_tensor[i])
                output, hidden = model(signal_tensor, input_tensor[i], hidden)
                # print(hidden.shape)
                # print(hidden)
                # print(output.shape)
                # print(output)
                # print(target_tensor[i])
                if output.topk(1)[1] == 5:
                    l = criterion(output, target_tensor[i])
                    loss += l
                    break
                l = criterion(output, target_tensor[i])
                # print(l)
                # if target_tensor[i] == 5:
                #     print("EOS")
                loss += l
                # print("-----------")
            loss.backward()

            for name, param in model.named_parameters():
                writer.add_histogram(name + '/gradients', param.grad, epoch * len(sound))

            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), epoch * len(sound))

            for name, param in model.named_parameters():
                writer.add_histogram(name, param.data, epoch * len(sound))

            
            # fixed += 1
        
        print(f"{timeSince(start)} (Epoch {epoch+1}/{num_epochs}), Loss: {loss.item()}")




if __name__ == "__main__":
    writer = SummaryWriter('!08_code/log/overfit-16th')
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    ANNOTATIONS_FILE = "dataset/00-very simple rhythm slowver/metadata.csv"
    AUDIO_DIR = "dataset/00-very simple rhythm slowver"
    SAMPLE_RATE = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64)

    sound = MusicDataset(ANNOTATIONS_FILE,
                    AUDIO_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    # NUM_SAMPLES,
                    rhythm_dict
                    )
    
    print(f"There are {len(sound)} samples in the dataset.")

    data_loader = DataLoader(sound)

    # print(data_loader)

    input_size = 42624
    hidden_size = 128
    output_size = len(rhythm_dict)
    max_length = 60


    model = RNN(input_size, hidden_size, output_size)

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    start = time.time()

    train(model, num_epochs, sound[4], criterion, optimizer, writer)
    torch.save(model.state_dict(), "!08_code/rnnnet.pth")
    print("Model trained and stored at rnnnet.pth")

    

