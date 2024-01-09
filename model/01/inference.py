import torch
import torch.nn as nn
from model import RNN
from preprocessing import MusicDataset
from torch.utils.data import Dataset, DataLoader
import torchaudio


def predict(model, input, n_output, target):
    model.eval()
    with torch.no_grad():
        seq = []
        output = model(input, n_output)
        output = output.reshape(n_output, -1)
        # print(output)
        for i in range(len(target)):
            out = output[i]
            topv, topi = out.topk(1)
            seq.append(topi.item())
    return seq, target



if __name__ == "__main__":

    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4}

    input_size = 1220
    hidden_size = 128
    output_size = len(rhythm_dict)

    rnn = RNN(input_size, hidden_size, output_size)
    state_dict = torch.load("!06_code/rnnnet.pth")
    rnn.load_state_dict(state_dict)

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
                    rhythm_dict
                    )
    
    random = torch.randint(0, 50, (1,)).item()
    input = sound[random][0]
    n_output = sound[random][1]
    target = sound[random][2]

    predicted, expected = predict(rnn, input, n_output, target)
    print(f"Predicted: '{list(map(rhythm_dict_swap.get, predicted))}'")
    print(f"Expected: '{list(map(rhythm_dict_swap.get, expected.tolist()))}'")