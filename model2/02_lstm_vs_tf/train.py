### Import Model -----------------------------###

# from model_lstm import EncoderDecoder
# from model_entf_delstm import EncoderDecoder
from model_tf import EncoderDecoder

### Import Fuctions and Class -----------------###
from preprocessing import MusicDataset, collate_fn
from inference import Inference
from inference_plot import InferencePlot

### Import Library -----------------------------###
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pathlib
import ast
import shutil

import pandas as pd
import numpy as np
import itertools
import time
import math
import matplotlib.pyplot as plt

###---------------------------------------------###

def timeSince(since): # Compute time
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def plot_confusion_matrix(cm, class_names):

    raw_cm = cm
    figure = plt.figure(figsize=(8, 8))
    
    # light mode
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1
    cm = np.around(cm.astype('float') / row_sums[:, np.newaxis], decimals=4) * 100
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, f"{(cm[i, j]):.2f}"+'%\n'+str(int(raw_cm[i, j])), horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def sequence_accuracy(pred, target):
    
    # Check for division by zero
    if len(target) == 0:
        return 0, 0, 0, 0
        
    # Length Accuracy (LA)
    acc_len = 0
    if len(pred) == len(target):
        acc_len = 1

    # Exact Order Accuracy (EOA)
    if len(pred) < len(target):
        pad_num = len(target) - len(pred)
        pred.extend([-1]*pad_num)
    
    acc_exactly = 0
    for i in range(len(target)):
        if pred[i] == target[i]:
            acc_exactly += 1
    acc_exactly = acc_exactly / len(target)
    
    acc_avg = (acc_exactly + acc_len) / 2

    return acc_exactly, acc_len, acc_avg

def train(model, num_epochs, train_set, criterion, 
          model_optimizer, validation_set, max_length, writer):
    
    # Setting before trainning -------------- #

    csv = {
        'target':[None],
        'predict':[None]
    }
    df = pd.DataFrame(csv)


    ### Trainning loop ------------------------ #

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        ### Create Batch ###
        train_set_shuffle = shuffle(train_set)
        validation_set_shuffle = shuffle(validation_set)

        batch_train = DataLoader(train_set_shuffle, batch_size=batch_size, collate_fn=collate_fn)
        batch_val = DataLoader(validation_set_shuffle, batch_size=batch_size, collate_fn=collate_fn)

        ### Mode train ###
        model.train()

        total_loss = []; total_EOA = []; total_LA = []; total_AvgAcc = []

        for batch in batch_train:
            signal_tensor, target_input, target_tensor = batch
            signal_tensor, target_input, target_tensor = signal_tensor.to(DEVICE), target_input.to(DEVICE), target_tensor.to(DEVICE)
            # mel_spectrogram = visualize_mel_spectrogram(signal_tensor, f'Mel Spectrogram - Epoch {epoch+1}')
            # writer.add_figure(f'Mel Spectrogram/{epoch * len(sound)}', mel_spectrogram, global_step=None, close=True, walltime=None)

            model_optimizer.zero_grad()

            decoder_outputs = model(signal_tensor, target=target_input)

            loss = criterion(
                    decoder_outputs.view(-1, output_size),
                    target_tensor.view(-1)
                    )

            loss.backward()

            model_optimizer.step()

            total_loss.append(loss.item())

            ## Calculate Accuracy ##
            for each in range(decoder_outputs.size(0)):
                decoder_output = decoder_outputs[each]
                predicted = []
                for elm in decoder_output:
                    topv, topi = elm.topk(1)
                    if topi.item() == rhythm_to_id["<EOS>"]:
                        predicted.append(ID_TO_RHYTHM[topi.item()])
                        break
                    predicted.append(ID_TO_RHYTHM[topi.item()])
                expected = list(map(ID_TO_RHYTHM.get, target_tensor[each].tolist()))
                expected = expected[:expected.index("<EOS>")+1]
                
                acc_eoa, acc_la, avg_acc = sequence_accuracy(predicted, expected)

                total_EOA.append(acc_eoa), total_LA.append(acc_la), total_AvgAcc.append(avg_acc)

                # If it's last epoch, save result in csv file #
                if epoch == num_epochs-1:
                    new_row = pd.DataFrame([{"target":f"{expected}", "predict":f"{predicted}"}])
                    df = pd.concat([df, new_row])
                    df.reset_index(drop=True, inplace=True)


        writer.add_scalar('Loss/train', sum(total_loss), epoch)
        writer.add_scalar('Avg.Loss/train', np.average(total_loss), epoch)   
        writer.add_scalar('Avg.Accuracy/train', np.average(total_AvgAcc), epoch)
        writer.add_scalar('EOA.Accuracy/train', np.average(total_EOA), epoch)
        writer.add_scalar('LA.Accuracy/train', np.average(total_LA), epoch)


        print(f"{timeSince(start)} (Epoch {epoch+1}/{num_epochs}), Loss: {sum(total_loss)}, Accuracy: {np.average(total_AvgAcc)}")

    
        ### Mode validate ###
        model.eval()

        with torch.no_grad():
            confmat = ConfusionMatrix(task="multiclass", num_classes=len(rhythm_to_id), ignore_index=-1).to(DEVICE)
            total_loss_val = []; total_EOA_val = []; total_LA_val = []; total_AvgAcc_val = []
            cf_matrix_all = np.zeros((len(rhythm_to_id),len(rhythm_to_id)))

            for batch in batch_val:
                signal_val, target_input_val, target_val = batch
                signal_val, target_input_val, target_val = signal_val.to(DEVICE), target_input_val.to(DEVICE), target_val.to(DEVICE)

                decoder_outputs_val = model(signal_val, max_loop=max_length)

                pad_target_val = torch.nn.functional.pad(target_val, (0, decoder_outputs_val.size(1)-target_val.size(1)), value=-1)

                cf_matrix = confmat(decoder_outputs_val.view(-1, output_size), pad_target_val.view(-1)).cpu().numpy()
                cf_matrix_all += cf_matrix

                loss_val = criterion(
                    decoder_outputs_val.view(-1, output_size),
                    pad_target_val.view(-1)
                    )
                
                total_loss_val.append(loss_val.item())

                for each_val in range(decoder_outputs_val.size(0)):
                    decoder_output_val = decoder_outputs_val[each_val]
                    predicted_val = []
                    for elm in decoder_output_val:
                        topv, topi = elm.topk(1)
                        if topi.item() == rhythm_to_id["<EOS>"]:
                            predicted_val.append(ID_TO_RHYTHM[topi.item()])
                            break
                        predicted_val.append(ID_TO_RHYTHM[topi.item()])
                    expected_val = list(map(ID_TO_RHYTHM.get, target_val[each_val].tolist()))
                    expected_val = expected_val[:expected_val.index("<EOS>")+1]

                    acc_exactly_val, acc_len_val, acc_avg_val = sequence_accuracy(predicted_val, expected_val)
                    total_EOA_val.append(acc_exactly_val), total_LA_val.append(acc_len_val), total_AvgAcc_val.append(acc_avg_val)

                    if epoch == num_epochs-1:
                        new_row = pd.DataFrame([{"target":f"{expected_val}", "predict":f"{predicted_val}"}])
                        df = pd.concat([df, new_row])
                        df.reset_index(drop=True, inplace=True)

            if (epoch+1)%25 == 0: # Save Cfm every 25 epochs
                writer.add_figure(f'Confusion matrix - Validate set', plot_confusion_matrix(cf_matrix_all, rhythm_to_id), epoch)


            print(f"Validate Set, Loss: {sum(total_loss_val)}, Accuracy: {np.average(total_AvgAcc_val)}")


            writer.add_scalar('Loss/validation', sum(total_loss_val), epoch)
            writer.add_scalar('Avg.Loss/validation', np.average(total_loss_val), epoch)      
            writer.add_scalar('Avg.Accuracy/validation', np.average(total_AvgAcc_val), epoch)
            writer.add_scalar('EOA.Accuracy/validation', np.average(total_EOA_val), epoch)
            writer.add_scalar('LA.Accuracy/validation', np.average(total_LA_val), epoch)


    ### Trainning Loop End Here ----------------------------------------- ##

    if experiment_name is not None:
        print(df)
        df.to_csv(f"{LOG_PATH}/train_test_predict.csv", index=False) 

    result_table = f"""
        | Metric | Value |
        |----------|-----------|
        | total loss / train | {sum(total_loss):.4f} |
        | avg loss / train | {np.average(total_loss):.4f} |
        | avg acc / train | {np.average(total_AvgAcc):.4f} |
        | total loss / validation | {sum(total_loss_val):.4f} |
        | avg loss / validation | {np.average(total_loss_val):.4f} |
        | avg acc / validation | {np.average(total_AvgAcc_val):.4f} |
    """
    result_table = '\n'.join(l.strip() for l in result_table.splitlines())
    writer.add_text("Result of Last Epoch", result_table, 0)

    print("end")


### Use for Logging ----------------------------------------------------###

def plot_histrogram(name, label):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    all_label = [j for i in label for j in ast.literal_eval(i)]
    plt.hist(all_label, density = True, width=0.5)
    plt.xticks([i for i in rhythm_to_id.keys()])
    ax.set_axisbelow(True)
    ax.grid()
    return fig

def visualize_data(idx_train, idx_val, train_dir, val_dir, writer):
    # train set
    train_label = pd.read_csv(train_dir+"/metadata.csv")
    train_label = train_label.loc[idx_train.tolist()]["rhythm_label"].to_list()
    writer.add_figure(f'Histrogram Train {train_dir}', plot_histrogram(f"{train_dir}", train_label), 0)

    # test set
    test_label = pd.read_csv(val_dir+"/metadata.csv")
    test_label = test_label.loc[idx_val.tolist()]["rhythm_label"].to_list()
    writer.add_figure(f'Histrogram Test {val_dir}', plot_histrogram(f"{val_dir}", test_label), 0)

    dataset_detail = f"""train_dir: {train_dir}, idx_train: {idx_train},
    val_dir: {val_dir}, idx_val: {idx_val}"""

    dataset_detail = '\n'.join(l.strip() for l in dataset_detail.splitlines())
    writer.add_text("dataset_detail", dataset_detail, 0)



if __name__ == "__main__":

    ### ------------ Variable / Parameter ----------------- ###
    force_start = "YES"

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

    lr = 0.0005
    num_epochs = 1000

    rhythm_to_id = {"<SOS>":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "rest_quarter": 6, "<EOS>":7}
    ID_TO_RHYTHM = {v: k for k, v in rhythm_to_id.items()}
    audio_dir = "dataset/data4note"
    validation_dir = None
    experiment_name = "transformer"
    random_state = 0

    PARAMETER_DICT = {'rhythm_to_id':rhythm_to_id, 'ID_TO_RHYTHM':ID_TO_RHYTHM, 'audio_dir': audio_dir,
                      'SAMPLE_RATE': SAMPLE_RATE, 'n_fft':n_fft, 'hop_length': hop_length, 'n_mels':n_mels,
                      'time_length':time_length, 'overlap':overlap, 'max_length': max_length, 
                      'input_size':input_size, 'hidden_size':hidden_size}

    #------------------------------------------------------#

    # Default variable with condition #
    if validation_dir is None: 
        validation_dir = audio_dir 
        validate_size = 0.3
    if experiment_name is None: 
        experiment_name = "Temp"

    LOG_PATH = f'{str(CURRENT_PATH)}/log/{audio_dir}/{str(experiment_name)}{time.strftime("%Y%m%d-%H%M", time.localtime())}'
    WRITER = SummaryWriter(LOG_PATH)
    MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels = n_mels
        )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device {DEVICE}")

    train_sound = MusicDataset(audio_dir + "/metadata.csv",
                    audio_dir,
                    MEL_SPECTROGRAM,
                    SAMPLE_RATE,
                    time_length,
                    overlap,
                    rhythm_to_id,
                    max_length,
                    DEVICE,
                    WRITER
                    )
    if validation_dir != audio_dir:
        validate_sound = MusicDataset(validation_dir + "/metadata.csv",
                        validation_dir,
                        MEL_SPECTROGRAM,
                        SAMPLE_RATE,
                        time_length,
                        overlap,
                        rhythm_to_id,
                        max_length,
                        DEVICE,
                        WRITER
                        )
        train_set = DataLoader(train_sound, batch_size=batch_size, collate_fn=collate_fn)
        validation_set = DataLoader(validate_sound, batch_size=batch_size, collate_fn=collate_fn)
        idx_train, idx_validation = np.arange(len(train_sound)), np.arange(len(validate_sound))
    else:
        train_set, validation_set, idx_train, idx_validation = train_test_split(train_sound, np.arange(len(train_sound)), test_size=validate_size, random_state=random_state, shuffle=True)

    print(f"There are {len(train_sound)} samples in the dataset.")
    signal_shape = set()
    for i in range(len(train_sound)): 
        signal, target_onehot, label = train_sound[i]; signal_shape.add(signal.shape)

    visualize_data(idx_train, idx_validation, audio_dir, validation_dir, WRITER)

    output_dict = rhythm_to_id
    output_size = len(rhythm_to_id)

    model = EncoderDecoder(input_size, hidden_size, output_dict, max_length).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scalars_loss_acc = {
            "loss-acc": {
                "total loss": ["Multiline", ["Loss/train", "Loss/validation"]],
                "avg loss": ["Multiline", ["Avg.Loss/train", "Avg.Loss/validation"]],
                "avg accuracy": ["Multiline", ["Avg.Accuracy/train", "Avg.Accuracy/validation"]],
                "EOA": ["Multiline", ["EOA.Accuracy/train", "EOA.Accuracy/validation"]],
                "LA": ["Multiline", ["LA.Accuracy/train", "LA.Accuracy/validation"]],
            },
        }
    WRITER.add_custom_scalars(scalars_loss_acc)

    parameter_table = f"""
        | Parameter | Value |
        |----------|-----------|
        | unique signal shape | {signal_shape} |
        | time_length | {time_length} |
        | overlap | {overlap} |
        | batch_size | {batch_size} |
        | n_fft | {n_fft} |
        | hop_length | {hop_length} |
        | n_mels | {n_mels} |
        | input_size | {input_size} |
        | hidden_size | {hidden_size} |
        | num_epochs | {num_epochs} |
        | learning_rate | {lr} |
    """
    parameter_table = '\n'.join(l.strip() for l in parameter_table.splitlines())
    WRITER.add_text("table", parameter_table, 0)

    print("Recheck Your Parameter")
    print(parameter_table)
    print("You want to continue?")
    force_start = str(input("YES|NO : ")) if force_start is None else "YES"
    if (force_start == "YES" ):
        print("OKiee, about to start...")

        start = time.time()

        train(model, num_epochs, train_set, criterion, model_optimizer, validation_set, max_length, WRITER)
        
        torch.save(model.state_dict(), f"{LOG_PATH}/model.pth")
        print("Model trained and stored at model.pth")

        print(Inference(audio_dir, PARAMETER_DICT, LOG_PATH))
        print(InferencePlot(LOG_PATH, rhythm_to_id, WRITER))
    else:
        shutil.rmtree(LOG_PATH)
        
