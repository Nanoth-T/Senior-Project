### Import Model -----------------------------###

# from model_lstm import EncoderDecoder
# from model_tf_linear import EncoderDecoder
from model import TransformerConv2dModel, TransformerEncoderLinear, LSTMModel

### Import Fuctions and Class -----------------###
from preprocessing import MusicDataset, custom_collate_fn
from training_loop import sequence_accuracy
import sys, os
sys.path.append(os.path.abspath('/home/bev/miniconda3/envs/myenv1/0rhythm_tempo_pj2/midi_processor'))
from processor import encode_midi, decode_midi # type: ignore

### Import Library -----------------------------###
import torch
import torchaudio
from torch.utils.data import DataLoader
import pathlib
import pandas as pd
import numpy as np
import csv

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

    total_acc_all = []
    total_acc_time = []
    total_acc_rhythm = []

    row = 0
    for i in sound_data:
        # print(f"sound{i}------------------------------------------")
        signal = i[0]
        target = i[2].tolist()[0]
        # signal = signal.unsqueeze(0)

        predicted  = predict(model, signal, max_length, all_note)
        decode_predicted = decode_midi(predicted)

        for i in decode_predicted.instruments:
            decode_predicted_list = []
            for note in i.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end)
                decode_predicted_list.append(rhythm)


        expected = [all_note[i] for i in target]
        expected = expected[:-1]

        decode_expected = decode_midi(expected)

        for j in decode_expected.instruments:
            decode_expected_list = []
            for note in j.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end)
                decode_expected_list.append(rhythm)

        # if decode_predicted_list != decode_expected_list:
        #     print(f"Predicted {len(decode_predicted_list)} items: {decode_predicted_list}")
        #     print(f"Expected {len(decode_expected_list)} items: {decode_expected_list}")

        data.append([expected, predicted, decode_expected_list, decode_predicted_list])
        pred, target = predicted.copy(), expected.copy()
        acc_all, acc_time = sequence_accuracy(pred, target)
        acc_rhythm = rhythm_accuracy(rhythm_predict=decode_predicted_list, rhythm_target=decode_expected_list)
        total_acc_all.append(acc_all)
        total_acc_time.append(acc_time)
        total_acc_rhythm.append(acc_rhythm)
        row += 1


    print(f"Accuracy All: {np.average(total_acc_all):.4%}, Accuracy Time: {np.average(total_acc_time):.4%}, Accuracy Rhythm: {np.average(total_acc_rhythm):.4%}")

    if wandb != None:
        wandb.log({'Avg.AccuracyAll/test': np.average(total_acc_all)})
        wandb.log({'Avg.AccuracyTime/test': np.average(total_acc_time)})
        wandb.log({'Avg.AccuracyRhythm/test': np.average(total_acc_rhythm)})


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
    

def calculate_rhythm_note(start_time, end_time, tempo=120):
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

    # rhythm_to_id = {"<SOS>":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "<PAD>": 6, "<EOS>":7}
    # ID_TO_RHYTHM = {v: k for k, v in rhythm_to_id.items()}
    addition = {
        "audio_dir" : "dataset/sample_audio3_30note",
        'metadata_path' : "dataset/sample_audio3_30note/metadata_chunks.csv",
        'file_name' : "sampleaudio3_longlength_chunk.csv",
    }
    
    CURRENT_PATH = pathlib.Path(__file__).parent.resolve()

    LOG_PATH = f"08_midilike/log/dataset/long_length_data/transformer_linear_dataset/long_length_data_220241001-2305"

    state_dict = torch.load(f"{LOG_PATH}/model.pth")
    config = state_dict["config_dict"]
    model = None

    print(Inference(model, config, LOG_PATH, addition))


