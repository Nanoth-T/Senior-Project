import torch
import torch.nn as nn
from model import RNN
from preprocessing import MusicDataset
from torch.utils.data import Dataset, DataLoader
import torchaudio


def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        seq = []
        output = model(input)
        # output = output.reshape(n_output, -1)
        # print(output)
        output = output.reshape(output.shape[1], -1)
        for i in range(output.shape[0]):
            out = output[i]
            topv, topi = out.topk(1)
            if topi == 5:
                seq.append(topi.item())
                break
            seq.append(topi.item())
        # for i in range(len(target)):
        #     out = output[i]
        #     topv, topi = out.topk(1)
        #     seq.append(topi.item())
    return seq, target



if __name__ == "__main__":

    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5}

    input_size = 1220
    hidden_size = 128
    output_size = len(rhythm_dict)

    rnn = RNN(input_size, hidden_size, output_size)
    state_dict = torch.load("!06_code ver2/rnnnet.pth")
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
    
    random = torch.randint(0, len(sound), (1,)).item()
    input = sound[random][0]
    target = sound[random][1]

    predicted, expected = predict(rnn, input, target)
    print(f"Predicted {len(predicted)} items: {list(map(rhythm_dict_swap.get, predicted))}")
    print(f"Expected {len(expected.tolist())} items: {list(map(rhythm_dict_swap.get, expected.tolist()))}")