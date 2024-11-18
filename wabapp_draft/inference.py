### Import Model -----------------------------###

# from model_lstm import EncoderDecoder
# from model_tf_linear import EncoderDecoder
from model import TransformerConv2dModel, TransformerEncoderLinear, LSTMModel

### Import Fuctions and Class -----------------###
from preprocessing import MusicDataset_Chunk, MusicDataset_Unchunk, custom_collate_fn
from training_loop import sequence_accuracy
from inference_plot import InferencePlot_Chunk, InferencePlot_Unchunk
import sys, os
sys.path.append(os.path.abspath('/home/bev/nanoth/0rhythm_tempo_pj2/midi_processor_mod'))
from processor import encode_midi, decode_midi # type: ignore

### Import Library -----------------------------###
import torch
import torchaudio
from torch.utils.data import DataLoader
import pathlib
import pandas as pd
import numpy as np
import csv
import wandb
import ast
# from deeprhythm import DeepRhythmPredictor


###---------------------------------------------###

def Inference_Chunk(model, config, LOG_PATH, addition = None, wandb = None):
    if addition != None:
        audio_dir = addition["audio_dir"]
        metadata_path = addition["metadata_path"]
        file_name = addition["file_name"]
    else:
        audio_dir = config['audio_dir']
        metadata_path = config['test_metadata']
        file_name = "test_data.csv"

    # global find_tempo_from_wav
    # find_tempo_from_wav = DeepRhythmPredictor()

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

    sound = MusicDataset_Chunk(annotations_file = metadata_path,
                         audio_dir = audio_dir,
                         transformation = MEL_SPECTROGRAM,
                         target_sample_rate = config['sample_rate'],
                         target_hop_length = config['hop_length'],
                         chunk_size = config['chunk_size'],
                         output_dict = config['output_dict'],
                         max_length = config['max_length'],
                         model_name = config['model_name'],
                         device = DEVICE)
    
    sound_data = DataLoader(sound, batch_size=1, collate_fn=custom_collate_fn(output_dict))
    test_full_data = pd.read_csv(metadata_path)

    data = []

    total_acc_all = []
    total_acc_time = []
    total_acc_rhythm = []

    row = 0
    for i, sound in enumerate(sound_data):
        # print(f"sound{i}------------------------------------------")
        signal = sound[0]
        metadata = sound[3]
        target = metadata[0]["midi_label"]
        
        tempo = metadata[0]["tempo"]
        # tempo = 120
        predicted  = predict(model, signal, max_length, all_note)
        decode_predicted = decode_midi(predicted)

        for i in decode_predicted.instruments:
            decode_predicted_list = []
            for note in i.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end, tempo=tempo)
                decode_predicted_list.append(rhythm)

        expected = [all_note[i] for i in target]
        decode_expected = decode_midi(expected)

        for j in decode_expected.instruments:
            decode_expected_list = []
            for note in j.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end, tempo=tempo)
                decode_expected_list.append(rhythm)

        # if decode_predicted_list != decode_expected_list:
        #     print(f"Predicted {len(decode_predicted_list)} items: {decode_predicted_list}")
        #     print(f"Expected {len(decode_expected_list)} items: {decode_expected_list}")

        data.append([expected, predicted, decode_expected_list, decode_predicted_list, str(metadata[0]["audio_idx"])])
        pred, target = predicted.copy(), expected.copy()
        acc_all, acc_time = sequence_accuracy(pred, target)
        acc_rhythm = rhythm_accuracy(rhythm_predict=decode_predicted_list, rhythm_target=decode_expected_list)
        total_acc_all.append(acc_all)
        total_acc_time.append(acc_time)
        total_acc_rhythm.append(acc_rhythm)
        row += 1
        # print(decode_predicted_list)

        # print(f"Accuracy: {acc_all:.2%}")


    print(f"Accuracy All: {np.average(total_acc_all):.4%}, Accuracy Time: {np.average(total_acc_time):.4%}, Accuracy Rhythm: {np.average(total_acc_rhythm):.4%}")

    if wandb != None:
        wandb.log({'Avg.AccuracyAll/test': np.average(total_acc_all)})
        wandb.log({'Avg.AccuracyTime/test': np.average(total_acc_time)})
        wandb.log({'Avg.AccuracyRhythm/test': np.average(total_acc_rhythm)})


    with open(f"{LOG_PATH}/{file_name}", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["encode_target", "encode_predict", "decode_target", "decode_predict", "audio_idx"])
        writer.writerows(data)

    

    full_data = combine_chunks(f"{LOG_PATH}/{file_name}", test_full_data, wandb)
    full_data.to_csv(f"{LOG_PATH}/{file_name.replace('.csv', '')}_FULL.csv", index=False)

    return "save file already."

def Inference_Unchunk(model, config, LOG_PATH, addition = None, wandb = None):

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

    sound = MusicDataset_Unchunk(annotations_file = metadata_path,
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
        info = i[3]
        tempo = int(info[0]["tempo"])
        # signal = signal.unsqueeze(0)
        # tempo = 120
        # print(signal)

        predicted  = predict(model, signal, max_length, all_note)
        # print(predicted)
        decode_predicted = decode_midi(predicted)

        for i in decode_predicted.instruments:
            decode_predicted_list = []
            for note in i.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end, tempo=tempo)
                decode_predicted_list.append(rhythm)


        expected = [all_note[i] for i in target]
        expected = expected[:-1]

        decode_expected = decode_midi(expected)

        for j in decode_expected.instruments:
            decode_expected_list = []
            for note in j.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end, tempo=tempo)
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

    if type(rhythm_predict) != list:
        rhythm_predict = ast.literal_eval(rhythm_predict)
    if type(rhythm_target) != list:
        rhythm_target = ast.literal_eval(rhythm_target)

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
    # print(start_time, end_time)
    duration_seconds = round(end_time - start_time, 2)

    beats_per_second = round(tempo / 60, 2)
    beats = round(duration_seconds * beats_per_second, 1)

    # print(duration_seconds, beats)
    
    # Determine the rhythm note based on the number of beats
    if beats > 3: # 4
        return "whole"
    elif 1.5 < beats <= 3: # 2
        return "half"
    elif 0.75 < beats <= 1.5: # 1
        return "quarter"
    elif 0.375 < beats <= 0.75: # 0.5
        return "8th"
    elif 0.25 < beats <= 0.375: # 0.25
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
            elif topi.item() == all_note.index("<PAD>"):
                break
            predicted.append(all_note[topi.item()])
        print(predicted)


    return predicted

def combine_chunks(csv_file, test_full_data, wandb=None):
    chunks = pd.read_csv(csv_file)
    for col in ['encode_target', 'encode_predict', 'decode_target', 'decode_predict']:
        chunks[col] = chunks[col].str.replace(", '<PAD>'", "")
        chunks[col] = chunks[col].str.replace(", '<EOS>'", "")
        chunks[col] = chunks[col].apply(lambda x : ast.literal_eval(x))
    chunks = chunks.groupby(["audio_idx"], as_index = False, sort=False).agg('sum')
    print(chunks)

    total_acc_all, total_acc_time, total_acc_rhythm = find_acc(chunks, test_full_data)

    if wandb != None:
        wandb.log({'FullAudio.AccuracyAll/test': total_acc_all})
        wandb.log({'FullAudio.AccuracyTime/test': total_acc_time})
        wandb.log({'FullAudio.AccuracyRhythm/test': total_acc_rhythm})
    
    return chunks


def find_acc(filename, test_file):
    total_acc_all = []; total_acc_time = []; total_acc_rhythm = []
    for i in range(len(filename)):
        acc_all, acc_time = sequence_accuracy(filename.loc[i, "encode_predict"], test_file.loc[i, "encode_label"])
        acc_rhythm = rhythm_accuracy(filename.loc[i, "decode_predict"], test_file.loc[i, "decode_rhythm"])
        total_acc_all.append(acc_all); total_acc_time.append(acc_time); total_acc_rhythm.append(acc_rhythm)
    return f"{np.average(total_acc_all):.4%}", f"{np.average(total_acc_time):.4%}", f"{np.average(total_acc_rhythm):.4%}"
    # return total_acc_all, total_acc_time, total_acc_rhythm




if __name__ == "__main__":

    # rhythm_to_id = {"<SOS>":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "<PAD>": 6, "<EOS>":7}
    # ID_TO_RHYTHM = {v: k for k, v in rhythm_to_id.items()}
    addition = {
        "audio_dir" : "@wabapp/demo_realsomg_melody_fixed_velo",
        'metadata_path' : "@wabapp/demo_realsomg_melody_fixed_velo/_metadata.csv",
        'file_name' : "demo_realsomg_melody_fixed_velo.csv",
    }
    
    CURRENT_PATH = pathlib.Path(__file__).parent.resolve()

    LOG_PATH = f"@wabapp/log/transformer_linear_unchunk_midilabel_0_20241105-0026"

    state_dict = torch.load(f"{LOG_PATH}/model.pth")
    config = state_dict["config_dict"]
    model = None

    # wandb.init(entity='rhythm-tempo-project', project='random_velo', id='iojn3ibt', resume='must')

    print(Inference_Unchunk(model, config, LOG_PATH, addition, wandb=None))
    # output_dict = None
    # print(InferencePlot(LOG_PATH, output_dict, wandb))


