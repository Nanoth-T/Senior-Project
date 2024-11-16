import torch
import torch.nn as nn
from model import EncoderRNN, DecoderRNN
from preprocessing import MusicDataset
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchmetrics
from torch.nn.utils.rnn import pad_sequence
import pathlib

def input_tensor(rt):
    input = torch.zeros(1, 1, len(rhythm_dict))
    input[0][0][rhythm_dict[rt]] = 1
    return input

def predict(encoder, decoder, signal, max_length):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(signal)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, max_loop=max_length)

        decoder_output = decoder_outputs[0]
        predicted = []
        for elm in decoder_output:
            topv, topi = elm.topk(1)
            if topi.item() == rhythm_dict["EOS"]:
                predicted.append('EOS')
                break
            predicted.append(rhythm_dict_swap[topi.item()])
    return predicted


def sequence_accuracy(pred, target):

    min_len = min(len(pred), len(target))
    min_len = min_len - 1

    correct = 0
    for i in range(min_len):
        if pred[i+1] == target[i+1]:
            correct += 1
    
    if len(pred)-1 == 0:
        return 0
    else:
        # Calculate accuracy
        accuracy = correct / (len(pred)-1)

    return accuracy


if __name__ == "__main__":

    current_path = pathlib.Path(__file__).parent.resolve()
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    AUDIO_DIR = "dataset/02/10"
    ANNOTATIONS_FILE = AUDIO_DIR + "/metadata.csv"
    SAMPLE_RATE = 22050
    time_length = 300
    max_length = 60
    batch_size = 10
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

    input_size = n_mels*time_length
    hidden_size = 128
    output_size = len(rhythm_dict)
    max_length = 60
    
    encoder = EncoderRNN(input_size, hidden_size)
    decoder = DecoderRNN(hidden_size, output_size, max_length)
    enstate_dict = torch.load(f"{str(current_path)}/enmodel.pth")
    destate_dict = torch.load(f"{str(current_path)}/demodel.pth")
    encoder.load_state_dict(enstate_dict)
    decoder.load_state_dict(destate_dict)

    # sound = []
    # sound.append((train_batch10, tg_batch10, lb_batch10))

    # select = 0


    sum_acc = 0
    for i in range(len(sound)):
        print(f"sound{i}------------------------------------------")
        signal = sound[i][0]
        target_onehot = input_tensor("SOS")
        target = sound[i][2]
        signal = signal.unsqueeze(0)

        predicted  = predict(encoder, decoder, signal, max_length)
        expected = list(map(rhythm_dict_swap.get, (target.tolist())))
        # expected.insert(0, "SOS")

        accuracy = sequence_accuracy(predicted, expected)
        sum_acc += accuracy
        if predicted != expected:
            print(f"Predicted {len(predicted)} items: {predicted}")
            print(f"Expected {len(expected)} items: {expected}")
            print(f"Accuracy: {accuracy:.2%}")
        else:
            print(f"Accuracy: {accuracy:.2%}")
    print(f"All Accuracy: {sum_acc/len(sound):.2%}")
