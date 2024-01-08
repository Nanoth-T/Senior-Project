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

# def train(model, num_epochs, sound, criterion, optimizer):
#     # Training loop
#     for epoch in range(num_epochs):
#         print(f"Epoch [{epoch+1}/{num_epochs}]")
#         for signal_tensor, input_tensor, target_tensor in sound:
#             optimizer.zero_grad()
#             loss = torch.Tensor([0])
#             h = None
#             for i in range(input_tensor.size(0)):
#                 output, h = model(signal_tensor, h)
#                 l = criterion(output, target_tensor[i])
#                 loss += l

#             loss.backward()


def train(rnn, signal_tensor, input_tensor, target_tensor):
    # print(signal_tensor.shape, input_tensor.shape)
    target_tensor.unsqueeze_(-1)
    # hidden = rnn.initHidden()
    rnn.zero_grad()

    loss = torch.Tensor([0]) # you can also just simply use ``loss = 0``
    output = rnn(signal_tensor, len(target_tensor))
    output = output.reshape(len(target_tensor), -1)
    # print(output)
    for i in range(len(target_tensor)):
        out = output[i]
        # topv, topi = out.topk(1)
        # print(topv, topi)
        l = criterion(out.reshape(1, 6), target_tensor[i])
        loss += l
    # print(l)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_tensor.size(0)


if __name__ == "__main__":
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS": 5}
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
                    NUM_SAMPLES,
                    rhythm_dict
                    )
    
    print(f"There are {len(sound)} samples in the dataset.")

    data_loader = DataLoader(sound)

    print(data_loader)

    input_size = 1188
    hidden_size = 128
    output_size = len(rhythm_dict)


    model = RNN(input_size, hidden_size, output_size)

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    learning_rate = 0.01

    n_iters = 10
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every ``plot_every`` ``iters``

    start = time.time()

    for iter in range(1, n_iters + 1):
        temp_loss = 0
        for signal_tensor, input_tensor, label_tensor in sound:
            output, loss = train(model, signal_tensor, input_tensor, label_tensor)
            temp_loss += loss

        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, temp_loss))
        total_loss += temp_loss
    
    torch.save(model.state_dict(), "06_code/rnnnet.pth")
    print("Model trained and stored at rnnnet.pth")

    

