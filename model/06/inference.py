import torch
import torch.nn as nn
from model import Model
from preprocessing import MusicDataset
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchmetrics

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

def sequence_accuracy(pred, target):

    min_len = min(len(pred), len(target))
    min_len = min_len - 1

    correct = 0
    for i in range(min_len):
        if pred[i+1] == target[i+1]:
            correct += 1

    # Calculate accuracy
    accuracy = correct / (len(pred)-1)

    return accuracy


if __name__ == "__main__":

    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6}

    input_size = 256*480
    hidden_size = 128
    output_size = len(rhythm_dict)

    rnn = Model(input_size, hidden_size, output_size)
    state_dict = torch.load("!10_code/model.pth")
    rnn.load_state_dict(state_dict)

    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    ANNOTATIONS_FILE = "dataset/02-rhythm/metadata.csv"
    AUDIO_DIR = "dataset/02-rhythm"
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
    
    # random = torch.randint(0, len(sound), (1,)).item()
    # # random = 0
    # signal = sound[random][0]
    # input = input_tensor("SOS")
    # target = sound[random][2]
    max_length = 60

    sum_acc = 0
    for i in range(len(sound)):
        signal = sound[i][0]
        input = input_tensor("SOS")
        target = sound[i][2]

        predicted, expected = predict(rnn, signal, input, target, max_length)
        expected = list(map(rhythm_dict_swap.get, (expected.tolist())))
        expected.insert(0, "SOS")

        accuracy = sequence_accuracy(predicted, expected)
        sum_acc += accuracy
        if predicted != expected:
            print(f"Predicted {len(predicted)} items: {predicted}")
            print(f"Expected {len(expected)} items: {expected}")
            print(f"Accuracy: {accuracy:.2%}")
    print(f"All Accuracy: {sum_acc/len(sound):.2%}")