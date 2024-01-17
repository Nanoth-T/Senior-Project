import torch
import torch.nn as nn
from model import Model
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
        # hidden = model.initHidden()
        seq.append("SOS")
        initstate = None
        for i in range(max_length):
            output, initstate = model(signal, input[0], initstate)
            topv, topi = output.topk(1)
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

    input_size = 333*256
    hidden_size = 128
    output_size = len(rhythm_dict)

    rnn = Model(input_size, hidden_size, output_size)
    state_dict = torch.load("!09_code/model.pth")
    rnn.load_state_dict(state_dict)

    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    ANNOTATIONS_FILE = "dataset/00-very simple rhythm slowver/metadata.csv"
    AUDIO_DIR = "dataset/00-very simple rhythm slowver"
    SAMPLE_RATE = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 2048,
        hop_length = 1025,
        n_mels = 256
        )

    sound = MusicDataset(ANNOTATIONS_FILE,
                    AUDIO_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    rhythm_dict
                    )
    
    random = torch.randint(0, len(sound), (1,)).item()
    random = 0
    signal = sound[random][0]
    input = input_tensor("SOS")
    target = sound[random][2]
    max_length = 60

    predicted, expected = predict(rnn, signal, input, target, max_length)
    print(f"Predicted {len(predicted)} items: {predicted}")
    expected = list(map(rhythm_dict_swap.get, (expected.tolist())))
    expected.insert(0, "SOS")
    print(f"Expected {len(expected)} items: {expected}")