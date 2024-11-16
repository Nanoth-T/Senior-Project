import torch
import torch.nn as nn
from model import RNN
from preprocessing import MusicDataset
from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
import math
from torch.utils.tensorboard import SummaryWriter


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(model, num_epochs, sound, criterion, optimizer, writer):
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        fixed = 0
        for signal_tensor, input_tensor, target_tensor in sound:

            # writer.add_images('Signal Spectrogram', signal_tensor.permute(1, 2, 0), epoch * len(sound))

            # if fixed == 1:
            #     break
            target_tensor.unsqueeze_(-1)
            hidden = model.initHidden()
            optimizer.zero_grad()
            loss = torch.Tensor([0])
            for i in range(input_tensor.size(0)):
                # print(input_tensor[i])
                output, hidden = model(signal_tensor, input_tensor[i], hidden)
                # print(output)
                l = criterion(output, target_tensor[i])
                # if target_tensor[i] == 5:
                #     print("EOS")
                loss += l
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
    writer = SummaryWriter('!08_code/log')
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    ANNOTATIONS_FILE = "dataset/01-simple rhythm/metadata.csv"
    AUDIO_DIR = "dataset/01-simple rhythm"
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

    print(data_loader)

    input_size = 78080
    hidden_size = 128
    output_size = len(rhythm_dict)


    model = RNN(input_size, hidden_size, output_size)

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    start = time.time()

    train(model, num_epochs, sound, criterion, optimizer, writer)
    torch.save(model.state_dict(), "!08_code/rnnnet.pth")
    print("Model trained and stored at rnnnet.pth")

    

