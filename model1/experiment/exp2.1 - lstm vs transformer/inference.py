### Import Model -----------------------------###

# from model_lstm import EncoderDecoder
# from model_tf_linear import EncoderDecoder
from model import TransformerConv2dModel, TransformerEncoderLinear, LSTMModel

### Import Fuctions and Class -----------------###
from preprocessing import MusicDataset, custom_collate_fn
from training_loop import sequence_accuracy, len_accuracy
from inference_plot import InferencePlot

### Import Library -----------------------------###
import torch
import torchaudio
from torch.utils.data import DataLoader
import pathlib
import pandas as pd
import numpy as np
import csv
import wandb

###---------------------------------------------###

def Inference(model, config, LOG_PATH, addition = None, wandb = None):

    if addition != None:
        audio_dir = addition["audio_dir"]
        metadata_path = addition["metadata_path"]
        file_name = addition["file_name"]
    else:
        audio_dir = config['audio_dir']
        metadata_path = config['test_metadata']
        file_name = "test_data.csv"


    output_dict = config['output_dict']
    all_note = output_dict["all_note"]
    print(all_note)

    max_length = config['max_length']

    MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
        sample_rate = config['sample_rate'],
        n_fft = config['n_fft'],
        hop_length = config['hop_length'],
        n_mels = config['n_mels']
        )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device {DEVICE}")
    
    state_dict = torch.load(f"{LOG_PATH}/model.pth")
    model = config['model']
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    sound = MusicDataset(annotations_file = metadata_path,
                         audio_dir = audio_dir,
                         transformation = MEL_SPECTROGRAM,
                         target_sample_rate = config['sample_rate'],
                         time_length = config['time_length'],
                         overlap = config['overlap'],
                         output_dict = config['output_dict'],
                         max_length = config['max_length'],
                         model_name = config['model_name'],
                         device = DEVICE)
    
    sound_data = DataLoader(sound, batch_size=1, collate_fn=custom_collate_fn(output_dict))

    data = []

    total_acc_all = []; total_acc_rhythm = []; total_acc_len = []

    row = 0
    for i in sound_data:
        # print(f"sound{i}------------------------------------------")
        signal = i[0]
        target = i[2].tolist()[0]
        # signal = signal.unsqueeze(0)

        predicted  = predict(model, signal, max_length, all_note)

        expected = [all_note[i] for i in target]
        expected = expected[:-1]


        # if predicted != expected:
        #     print(f"Predicted {len(predicted)} items: {predicted}")
        #     print(f"Expected {len(expected)} items: {expected}")

        data.append([expected, predicted, expected, predicted])
        pred, target = predicted.copy(), expected.copy()
        acc_len = len_accuracy(predicted, expected)
        acc_all = sequence_accuracy(pred, target)
        acc_rhythm = rhythm_accuracy(rhythm_predict=pred, rhythm_target=target)
        total_acc_all.append(acc_all)
        total_acc_rhythm.append(acc_rhythm)
        total_acc_len.append(acc_len)
        row += 1


    print(f"Accuracy All: {np.average(total_acc_all):.4%}, Accuracy Rhythm: {np.average(total_acc_rhythm):.4%}, Accuracy Length: {np.average(total_acc_len):.4%}")

    if wandb != None:
        wandb.log({'Avg.AccuracyAll/test': np.average(total_acc_all)})
        wandb.log({'Avg.AccuracyRhythm/test': np.average(total_acc_rhythm)})
        wandb.log({'Avg.AccuracyLength/test': np.average(total_acc_len)})

    with open(f"{LOG_PATH}/{file_name}", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["encode_target", "encode_predict", "decode_target", "decode_predict"])
        writer.writerows(data)

    return "save file already."

def rhythm_accuracy(rhythm_predict, rhythm_target):

    if len(rhythm_predict) == 0:
        return 0

    max_length = max(len(rhythm_predict), len(rhythm_target))
    if len(rhythm_predict) != len(rhythm_target):
        rhythm_predict.extend(["<PAD>"]*(max_length-len(rhythm_predict)))
        rhythm_target.extend(["<PAD>"]*(max_length-len(rhythm_target)))

    acc_rhythm = 0
    for i in range(max_length):
        if rhythm_predict[i] == rhythm_target[i]:
            acc_rhythm += 1
    
    acc_rhythm = acc_rhythm / max_length
    return acc_rhythm

def calculate_rhythm_note(start_time, end_time, tempo):
    duration_seconds = end_time - start_time
    beats_per_second = tempo / 60
    beats = duration_seconds * beats_per_second
    
    # Determine the rhythm note based on the number of beats
    if beats >= 4:
        return "whole"
    elif beats >= 2:
        return "half"
    elif beats >= 1:
        return "quarter"
    elif beats >= 0.5:
        return "8th"
    elif beats >= 0.25:
        return "16th"
    else:
        return "16th"

def predict(model, signal, max_length, all_note):
 
    model.eval()
    with torch.no_grad():

        decoder_outputs = model(signal, max_loop=max_length)
        decoder_output = decoder_outputs[0]

        predicted = []
        for elm in decoder_output:
            topv, topi = elm.topk(1)
            if topi.item() == all_note.index("<EOS>"):
                break
            predicted.append(all_note[topi.item()])
        # print(predicted)


    return predicted




if __name__ == "__main__":

    addition = {
        "audio_dir" : "dataset/diff_5to7note_data",
        'metadata_path' : "dataset/diff_5to7note_data/test_metadata.csv",
        'file_name' : "test_data.csv",
    }
    
    CURRENT_PATH = pathlib.Path(__file__).parent.resolve()

    LOG_PATH = f"00_main/log/dataset/diff_5to7note_data/lstm_dataset/diff_5to7note_data_220240927-1959"

    state_dict = torch.load(f"{LOG_PATH}/model.pth")
    config = state_dict["config_dict"]
    model = None
    output_dict =None

    wandb.init(entity='rhythm-tempo-project', project='exp1-lstm-vs-transformer-repeat', id='xvongyvm', resume='must')

    print(Inference(model, config, LOG_PATH, addition, wandb))
    print(InferencePlot(LOG_PATH, output_dict, wandb))
