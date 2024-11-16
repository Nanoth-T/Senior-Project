import torch
import torch.nn as nn
from model import RNN
from preprocessing import MusicDataset
from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(model, num_epochs, sound, criterion, optimizer):
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for signal_tensor, target_tensor in sound:
            optimizer.zero_grad()
            loss = torch.Tensor([0])
            # output = model(signal_tensor, n_output)
            output = model(signal_tensor)
            # print(target_tensor)
            output = output.reshape(output.shape[1], -1)
            # print(output.shape)
            for i in range(output.shape[0]):
                # print(i)
                out = output[i]
                # print(out.shape)
                topv, topi = out.topk(1)
                # print(topi)
                # print(target_tensor.unsqueeze(-1)[i].shape)
                if topi == 5:
                    l = criterion(out, target_tensor[i])
                    loss += l
                    break
                # print(len(target_tensor))
                l = criterion(out, target_tensor[i])
                loss += l
                if i >= len(target_tensor)-1:
                    break
            loss.backward()
            optimizer.step()
        
        print(f"{timeSince(start)} (Epoch {epoch+1}/{num_epochs}), Loss: {loss.item()}")

if __name__ == "__main__":
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    ANNOTATIONS_FILE = "dataset/01-simple rhythm/metadata.csv"
    AUDIO_DIR = "dataset/01-simple rhythm"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 607744

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

    input_size = 1220
    hidden_size = 128
    output_size = len(rhythm_dict)


    model = RNN(input_size, hidden_size, output_size)

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    start = time.time()

    train(model, num_epochs, sound, criterion, optimizer)
    torch.save(model.state_dict(), "!06_code ver2/rnnnet.pth")
    print("Model trained and stored at rnnnet.pth")

    

