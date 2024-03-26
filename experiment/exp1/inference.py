import torch
import torch.nn as nn
from model import EncoderRNN, DecoderRNN, EncoderDecoder
from preprocessing import MusicDataset, create_data_loader
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchmetrics
from torch.nn.utils.rnn import pad_sequence
import pathlib
import pandas as pd
import numpy as np

def Inference(AUDIO_DIR, TEST_DIR, time_length, max_length, n_mels, rhythm_dict, path):
    # current_path = pathlib.Path(__file__).parent.resolve()
    # rhythm_dict = {"SOS":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "rest_quarter": 6, "EOS":7}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    ANNOTATIONS_FILE = TEST_DIR + "/metadata.csv"
    SAMPLE_RATE = 22050
    #n_mels = 512

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 2048,
        # hop_length = 1025,
        n_mels = n_mels
        )
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    sound = MusicDataset(ANNOTATIONS_FILE,
                    TEST_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    rhythm_dict,
                    max_length,
                    device
                    )

    input_size = n_mels*time_length
    hidden_size = 128
    output_dict = rhythm_dict
    output_size = len(rhythm_dict)
    
    ende_model = EncoderDecoder(input_size, hidden_size, output_dict, max_length).to(device)
    endestate_dict = torch.load(f"{path}/last_ende_model.pth")
    # enstate_dict = torch.load(f"{str(current_path)}/enmodel.pth")
    # destate_dict = torch.load(f"{str(current_path)}/demodel.pth")
    ende_model.load_state_dict(endestate_dict)

    # sound = []
    # sound.append((train_batch10, tg_batch10, lb_batch10))

    sound_data = create_data_loader(sound, batch_size=1)

    csv = {
        'file name':["empty"],
        'target':["empty"],
        'predict':["empty"]
    }
    df = pd.DataFrame(csv)

    # select = 0
    all_acc_exactly = []
    all_acc_len = []
    all_acc_in = []
    all_acc_avg = []

    for i in range(len(sound_data)):
        print(f"sound{i}------------------------------------------")
        signal = sound[i][0]
        target = sound[i][2]
        signal = signal.unsqueeze(0)

        predicted  = predict(ende_model, signal, max_length, rhythm_dict, rhythm_dict_swap)
        expected = list(map(rhythm_dict_swap.get, (target.tolist())))
        # expected.insert(0, "SOS")

        if predicted != expected:
            print(f"Predicted {len(predicted)} items: {predicted}")
            print(f"Expected {len(expected)} items: {expected}")

        df.loc[i] = [f"sound{i}", expected, predicted]  
        pred = predicted.copy()
        target = expected.copy()
        acc_exactly, acc_len, acc_in, acc_avg = sequence_accuracy(pred, target)
        all_acc_exactly.append(acc_exactly)
        all_acc_len.append(acc_len)
        all_acc_in.append(acc_in)
        all_acc_avg.append(acc_avg)

        print(f"Accuracy: {acc_exactly:.2%}")

    print(f"All Accuracy Exactly: {np.average(all_acc_exactly):.4%}")
    print(f"All Accuracy Length: {np.average(all_acc_len):.4%}")
    print(f"All Accuracy is in: {np.average(all_acc_in):.4%}")
    print(f"All Accuracy Average: {np.average(all_acc_avg):.4%}")

    # print(df)
    df.to_csv(f"{path}/all_data.csv", index=False) 

    return "save file already."



def predict(ende_model, signal, max_length, rhythm_dict, rhythm_dict_swap):
 
    ende_model.eval()
    with torch.no_grad():
        decoder_outputs = ende_model(signal, max_loop=max_length)

        decoder_output = decoder_outputs[0]
        predicted = []
        # print(decoder_output)
        for elm in decoder_output:
            topv, topi = elm.topk(1)
            if topi.item() == rhythm_dict["EOS"]:
                predicted.append('EOS')
                break
            predicted.append(rhythm_dict_swap[topi.item()])
        print(predicted)
    return predicted


def sequence_accuracy(pred, target):
    
    # Check for division by zero
    if len(target) == 0:
        return 0, 0, 0, 0

    # In-Order Accuracy (IOA)
    temp_target = target.copy()
    acc_in = 0
    for op in pred:
        if op in temp_target:
            temp_target.remove(op)
            acc_in += 1
    acc_in = acc_in / len(target)
        
    # Length Accuracy (LA)
    acc_len = 0
    # print(f"pred:{pred}")
    # print(f"target{target}")
    if len(pred) == len(target):
        acc_len = 1

    # Exact Order Accuracy (EOA)
    elif len(pred) < len(target):
        pad_num = len(target) - len(pred)
        pred.extend([-1]*pad_num)
    elif len(pred) > len(target):
        pad_num = len(pred) - len(target)
        target.extend([-1]*pad_num)
    
    acc_exactly = 0
    for i in range(len(target)):
        if pred[i] == target[i]:
            acc_exactly += 1
    acc_exactly = acc_exactly / len(target)
    
    acc_avg = (acc_exactly + acc_len + acc_in) / 3

    return acc_exactly, acc_len, acc_in, acc_avg



if __name__ == "__main__":

    current_path = pathlib.Path(__file__).parent.resolve()
    rhythm_dict = {"SOS":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "rest_quarter": 6, "EOS":7}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    AUDIO_DIR = "dataset/data4note"
    TEST_DIR = "dataset/data4note"
    ANNOTATIONS_FILE = TEST_DIR + "/metadata.csv"
    SAMPLE_RATE = 22050
    time_length = 400
    max_length = 60
    batch_size = 1
    n_mels = 256
    #n_mels = 512

    experiment = f"exp_log/{AUDIO_DIR}_timelength400"
    # experiment = None

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 2048,
        # hop_length = 1025,
        n_mels = n_mels
        )
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    sound = MusicDataset(ANNOTATIONS_FILE,
                    TEST_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    rhythm_dict,
                    max_length,
                    device
                    )

    input_size = n_mels*time_length
    hidden_size = 128
    output_dict = rhythm_dict
    output_size = len(rhythm_dict)
    max_length = 60
    
    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_dict, max_length).to(device)
    enstate_dict = torch.load(f"{str(current_path)}/{experiment}/last_enmodel.pth")
    destate_dict = torch.load(f"{str(current_path)}/{experiment}/last_demodel.pth")
    # enstate_dict = torch.load(f"{str(current_path)}/enmodel.pth")
    # destate_dict = torch.load(f"{str(current_path)}/demodel.pth")
    encoder.load_state_dict(enstate_dict)
    decoder.load_state_dict(destate_dict)

    # sound = []
    # sound.append((train_batch10, tg_batch10, lb_batch10))

    sound_data = create_data_loader(sound, batch_size)

    csv = {
        'file name':["empty"],
        'target':["empty"],
        'predict':["empty"]
    }
    df = pd.DataFrame(csv)

    # select = 0
    all_acc_exactly = []
    all_acc_len = []
    all_acc_in = []
    all_acc_avg = []

    for i in range(len(sound_data)):
        print(f"sound{i}------------------------------------------")
        signal = sound[i][0]
        # print(signal.shape)
        target = sound[i][2]
        signal = signal.unsqueeze(0)

        predicted  = predict(encoder, decoder, signal, max_length)
        expected = list(map(rhythm_dict_swap.get, (target.tolist())))
        # expected.insert(0, "SOS")

        if predicted != expected:
            print(f"Predicted {len(predicted)} items: {predicted}")
            print(f"Expected {len(expected)} items: {expected}")

        df.loc[i] = [f"sound{i}", expected, predicted]  
        pred = predicted.copy()
        target = expected.copy()
        acc_exactly, acc_len, acc_in, acc_avg = sequence_accuracy(pred, target)
        all_acc_exactly.append(acc_exactly)
        all_acc_len.append(acc_len)
        all_acc_in.append(acc_in)
        all_acc_avg.append(acc_avg)

        print(f"Accuracy: {acc_exactly:.2%}")

    print(f"All Accuracy Exactly: {np.average(all_acc_exactly):.4%}")
    print(f"All Accuracy Length: {np.average(all_acc_len):.4%}")
    print(f"All Accuracy is in: {np.average(all_acc_in):.4%}")
    print(f"All Accuracy Average: {np.average(all_acc_avg):.4%}")

    # print(df)
    df.to_csv(f"{str(current_path)}/{experiment}/inference_data4note_model_data4note.csv", index=False) 

