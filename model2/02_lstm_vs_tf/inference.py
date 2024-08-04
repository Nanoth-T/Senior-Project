### Import Model -----------------------------###

# from model_lstm import EncoderDecoder
# from model_entf_delstm import EncoderDecoder
from model_tf import EncoderDecoder

### Import Fuctions and Class -----------------###
from preprocessing import MusicDataset, collate_fn

### Import Library -----------------------------###
import torch
import torchaudio
from torch.utils.data import DataLoader
import pathlib
import pandas as pd
import numpy as np

###---------------------------------------------###

def Inference(audio_dir, PARAMETER_DICT, LOG_PATH):
    rhythm_to_id = PARAMETER_DICT['rhythm_to_id']
    ID_TO_RHYTHM = PARAMETER_DICT['ID_TO_RHYTHM']

    SAMPLE_RATE, n_fft, hop_length, n_mels = PARAMETER_DICT['SAMPLE_RATE'], PARAMETER_DICT['n_fft'], PARAMETER_DICT['hop_length'], PARAMETER_DICT['n_mels']
    time_length, overlap, max_length = PARAMETER_DICT['time_length'], PARAMETER_DICT['overlap'], PARAMETER_DICT['max_length']
    input_size, hidden_size = PARAMETER_DICT['input_size'], PARAMETER_DICT['hidden_size']

    MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels = n_mels
        )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device {DEVICE}")

    output_dict = rhythm_to_id
    output_size = len(rhythm_to_id)
    
    model = EncoderDecoder(input_size, hidden_size, output_dict, max_length).to(DEVICE)
    state_dict = torch.load(f"{LOG_PATH}/model.pth")
    model.load_state_dict(state_dict)

    sound = MusicDataset(audio_dir + "/metadata.csv",
                        audio_dir,
                        MEL_SPECTROGRAM,
                        SAMPLE_RATE,
                        time_length,
                        overlap,
                        rhythm_to_id,
                        max_length,
                        DEVICE
                        )
    sound_data = DataLoader(sound, batch_size=1, collate_fn=collate_fn)

    csv = {
        'file name':["empty"],
        'target':["empty"],
        'predict':["empty"]
    }
    df = pd.DataFrame(csv)

    total_EOA = []
    total_LA = []
    total_AvgAcc = []

    for i in range(len(sound_data)):
        print(f"sound{i}------------------------------------------")
        signal = sound[i][0]
        target = sound[i][2]
        signal = signal.unsqueeze(0)

        predicted  = predict(model, signal, max_length, rhythm_to_id, ID_TO_RHYTHM)

        expected = list(map(ID_TO_RHYTHM.get, (target.tolist())))

        if predicted != expected:
            print(f"Predicted {len(predicted)} items: {predicted}")
            print(f"Expected {len(expected)} items: {expected}")

        df.loc[i] = [f"sound{i}", expected, predicted]  
        pred, target = predicted.copy(), expected.copy()
        acc_eoa, acc_la, avg_acc = sequence_accuracy(pred, target)
        total_EOA.append(acc_eoa)
        total_LA.append(acc_la)
        total_AvgAcc.append(avg_acc)

        print(f"Accuracy: {acc_eoa:.2%}")

    print(f"Average EOA: {np.average(total_EOA):.4%}")
    print(f"Average LA: {np.average(total_LA):.4%}")
    print(f"Average Accuracy: {np.average(total_AvgAcc):.4%}")

    df.to_csv(f"{LOG_PATH}/all_data.csv", index=False) 

    return "save file already."



def predict(model, signal, max_length, rhythm_to_id, id_to_rhythm):
 
    model.eval()
    with torch.no_grad():

        decoder_outputs = model(signal, max_loop=max_length)
        decoder_output = decoder_outputs[0]

        predicted = []
        for elm in decoder_output:
            topv, topi = elm.topk(1)
            if topi.item() == rhythm_to_id["<EOS>"]:
                predicted.append('<EOS>')
                break
            predicted.append(id_to_rhythm[topi.item()])
        print(predicted)

    return predicted


def sequence_accuracy(pred, target):
    
    # Check for division by zero
    if len(target) == 0:
        return 0, 0, 0, 0
        
    # Length Accuracy (LA)
    acc_la = 0
    if len(pred) == len(target):
        acc_la = 1

    # Exact Order Accuracy (EOA)
    elif len(pred) < len(target):
        pad_num = len(target) - len(pred)
        pred.extend([-1]*pad_num)
    
    acc_eoa = 0
    for i in range(len(target)):
        if pred[i] == target[i]:
            acc_eoa += 1
    acc_eoa = acc_eoa / len(target)
    
    avg_acc = (acc_eoa + acc_la) / 2

    return acc_eoa, acc_la, avg_acc



if __name__ == "__main__":

    rhythm_to_id = {"<SOS>":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "rest_quarter": 6, "<EOS>":7}
    ID_TO_RHYTHM = {v: k for k, v in rhythm_to_id.items()}
    audio_dir = "dataset/prelim_data"
    CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
    SAMPLE_RATE = 22050
    n_fft = 2048
    hop_length = 1025
    n_mels = 256

    time_length = 200
    overlap = 50
    max_length = 60
    batch_size = 10
    input_size = n_mels*time_length
    hidden_size = 128

    LOG_PATH = f"05_tfm_vs_lstm/rerun_exp_log/dataset/prelim_data_tf_3/log"

    PARAMETER_DICT = {
        'rhythm_to_id':rhythm_to_id, 'ID_TO_RHYTHM':ID_TO_RHYTHM, 'audio_dir': audio_dir,
                      'SAMPLE_RATE': SAMPLE_RATE, 'n_fft':n_fft, 'hop_length': hop_length, 'n_mels':n_mels,
                      'time_length':time_length, 'overlap':overlap, 'max_length': max_length, 
                      'input_size':input_size, 'hidden_size':hidden_size
    }

    print(Inference(audio_dir, PARAMETER_DICT, LOG_PATH))
