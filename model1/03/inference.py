import torch
import torch.nn as nn
from model import RNN
from preprocessing import MusicDataset
from torch.utils.data import Dataset, DataLoader
import torchaudio

def input_tensor(rt):
    input = torch.zeros(1, 1, len(rhythm_dict))
    input[0][0][rhythm_dict[rt]] = 1
    return input

def predict(model, signal, input, target, max_length):
    model.eval()
    with torch.no_grad():
        seq = []
        hidden = model.initHidden()
        seq.append("SOS")
        for i in range(max_length):
            output, hidden = model(signal, input[0], hidden)
            topv, topi = output.topk(1)
            # print(topi.item())
            if topi.item() == rhythm_dict["EOS"]:
                seq.append("EOS")
                break
            else:
                rt = rhythm_dict_swap[topi.item()]
                seq.append(rt)
            input = input_tensor(rt)
            # print(input)
        # for i in range(len(target)):
        #     out = output[i]
        #     topv, topi = out.topk(1)
        #     seq.append(topi.item())
    return seq, target



if __name__ == "__main__":

    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6}

    input_size = 78080
    hidden_size = 128
    output_size = len(rhythm_dict)

    rnn = RNN(input_size, hidden_size, output_size)
    state_dict = torch.load("!08_code/rnnnet.pth")
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
    # random = 0
    signal = sound[random][0]
    input = input_tensor("SOS")
    target = sound[random][2]
    max_length = 60

    predicted, expected = predict(rnn, signal, input, target, max_length)
    print(f"Predicted {len(predicted)} items: {predicted}")
    print(f"Expected {len(expected.tolist())} items: {list(map(rhythm_dict_swap.get, expected.tolist()))}")