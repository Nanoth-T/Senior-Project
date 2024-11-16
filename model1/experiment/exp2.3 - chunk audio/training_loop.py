### Import Model -----------------------------###

from model import TransformerConv2dModel

### Import Fuctions and Class -----------------###
from preprocessing import MusicDataset, custom_collate_fn
# from inference import Inference
# from inference_plot import InferencePlot

### Import Library -----------------------------###
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

import wandb

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pathlib
import ast
import shutil

import pandas as pd
from tabulate import tabulate
import numpy as np
import itertools
import time
import math
import matplotlib.pyplot as plt

###---------------------------------------------###
def run_training(model, train_set, validation_set, config, wandb):
    global batch_size, DEVICE, output_dict, start, experiment_name, LOG_PATH, max_length, all_note, n_note

    experiment_name = config['experiment_name']
    LOG_PATH = config['log_path']

    output_dict = config['output_dict']
    all_note = output_dict['all_note']
    n_note = output_dict['n_note']

    start = time.time()

    model = config['model']
    criterion = config['criterion']
    model_optimizer = config['model_optimizer']
    max_length = config['max_length']

    
    batch_size = config['batch_size']
    input_size = config['input_size']
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device {DEVICE}")


    lr = config['learning_rate']
    num_epochs = config['epochs']

    torch.cuda.empty_cache()
    train(model, num_epochs, train_set, criterion, model_optimizer, validation_set, max_length, wandb)
    torch.cuda.empty_cache()

    torch.save({
        'model_state_dict': model.state_dict(),
        'config_dict': config
        }, f"{LOG_PATH}/model.pth")
    # torch.save(model.state_dict(), f"{LOG_PATH}/model.pth")
    # wandb.log_artifact(f"{LOG_PATH}/model.pth", type='model')
    print("Model trained and stored at model.pth")

    




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

    RANGE_NOTE_ON = list(range(0, 128)) # 0-127
    RANGE_NOTE_OFF = list(range(128, 256))  # 128-255
    RANGE_TIME_SHIFT = list(range(256, 356))  # 256-355
    RANGE_VEL = list(range(356, 388))  # 356-387

    # print("pred:", pred)
    # print("target:", target)

    if len(pred) == 0:
        return 0, 0

    max_length = max(len(pred), len(target))
    if len(pred) != len(target):
        pred.extend(["<PAD>"]*(max_length-len(pred)))
        target.extend(["<PAD>"]*(max_length-len(target)))

    # Exact Accuracy (EOA)
    acc_all = 0
    acc_time = 0; cnt_time = 0
    for i in range(max_length):
        if target[i] in RANGE_TIME_SHIFT:
            cnt_time += 1
            if pred[i] == target[i]:
                acc_time += 1
        if pred[i] in RANGE_TIME_SHIFT and target[i] not in RANGE_TIME_SHIFT:
            acc_time -= 1
        if pred[i] == target[i]:
            acc_all += 1

    acc_all = acc_all / max_length
    acc_time = acc_time / cnt_time

    return acc_all, acc_time



def train(model, num_epochs, train_set, criterion, 
          model_optimizer, validation_set, max_length, wandb):
    
    # Setting before trainning -------------- #

    csv = {
        'target':[None],
        'predict':[None]
    }
    df = pd.DataFrame(csv)



    ### Trainning loop ------------------------ #

    for epoch in range(1, num_epochs+1):
        print(f"Epoch [{epoch}/{num_epochs}]")

        ### Create Batch ###
        train_set_shuffle = shuffle(train_set)
        validation_set_shuffle = shuffle(validation_set)

        batch_train = DataLoader(train_set_shuffle, batch_size=batch_size, collate_fn=custom_collate_fn(output_dict))
        batch_val = DataLoader(validation_set_shuffle, batch_size=batch_size, collate_fn=custom_collate_fn(output_dict))

        ### Mode train ###
        model.train()

        

        total_loss = []
        total_acc_all = []
        total_acc_time = []

        for batch in batch_train:
            signal_tensor, input_tensor, target_tensor = batch
            signal_tensor, input_tensor, target_tensor = signal_tensor.to(DEVICE), input_tensor.to(DEVICE), target_tensor.to(DEVICE)
            # mel_spectrogram = visualize_mel_spectrogram(signal_tensor, f'Mel Spectrogram - Epoch {epoch+1}')
            # wandb.add_figure(f'Mel Spectrogram/{epoch * len(sound)}', mel_spectrogram, global_step=None, close=True, walltime=None)

            model_optimizer.zero_grad()

            decoder_outputs = model(signal_tensor, target=input_tensor)

            loss = criterion(
                    decoder_outputs.view(-1, n_note),
                    target_tensor.view(-1)
                    )
            loss.backward()

            model_optimizer.step()

            total_loss.append(loss.item())

            ## Calculate Accuracy ##
            for each in range(decoder_outputs.size(0)):
                decoder_output = decoder_outputs[each]
                predicted = []
                for i in decoder_output:
                    _, topi = i.topk(1)
                    pred = all_note[topi.item()]
                    predicted.append(pred)
                expected = [all_note[i] for i in target_tensor[each]]
                
                acc_all, acc_time = sequence_accuracy(predicted, expected)

                total_acc_all.append(acc_all)
                total_acc_time.append(acc_time)

                # If it's last epoch, save result in csv file #
                if epoch == num_epochs-1:
                    new_row = pd.DataFrame([{"target":f"{expected}", "predict":f"{predicted}"}])
                    df = pd.concat([df, new_row])
                    df.reset_index(drop=True, inplace=True)

        if epoch%10 == 0:
            wandb.log({"epoch": epoch, 'Loss/train': np.average(total_loss)})
            wandb.log({"epoch": epoch, 'AccuracyAll/train': np.average(total_acc_all)})
            wandb.log({"epoch": epoch, 'AccuracyTime/train': np.average(total_acc_time)})

        print(f"{timeSince(start)} (Epoch {epoch}/{num_epochs}), Loss: {np.average(total_loss):.4}, Accuracy All: {np.average(total_acc_all):.4}, , Accuracy Time: {np.average(total_acc_time):.4}")

    
        ### Mode validate ###
        model.eval()

        with torch.no_grad():
            # confmat = ConfusionMatrix(task="multiclass", num_classes=len(rhythm_to_id), ignore_index=-1).to(DEVICE)
            total_loss_val = []
            total_acc_all_val=[]
            total_acc_time_val=[]
            # cf_matrix_all = np.zeros((len(rhythm_to_id),len(rhythm_to_id)))

            for batch in batch_val:
                signal_tensor_val, input_tensor_val, target_tensor_val = batch
                signal_tensor_val, input_tensor_val, target_tensor_val = signal_tensor_val.to(DEVICE), input_tensor_val.to(DEVICE), target_tensor_val.to(DEVICE)

                decoder_output_vals = model(signal_tensor_val, max_loop=max_length)
                
                if decoder_output_vals.size(1) >= target_tensor_val.size(1):
                    padd_num = decoder_output_vals.size(1) - target_tensor_val.size(1)
                    target_val = torch.nn.functional.pad(target_tensor_val, (0, padd_num), mode='constant', value=all_note.index("<PAD>"))
                    prediction_val = decoder_output_vals
                elif decoder_output_vals.size(1) < target_tensor_val.size(1):
                    padd_num = target_tensor_val.size(1) -  decoder_output_vals.size(1)
                    prediction_val = torch.nn.functional.pad(decoder_output_vals, (0, 0, 0, padd_num), mode='constant', value=all_note.index("<PAD>"))
                    target_val = target_tensor_val
                # cf_matrix = confmat(decoder_outputs_val.view(-1, output_size), pad_target_val.view(-1)).cpu().numpy()
                # cf_matrix_all += cf_matrix
                
                loss_val = criterion(prediction_val.view(-1, n_note), 
                                     target_val.view(-1))
                
                total_loss_val.append(loss_val.item())
                
                ## Calculate Accuracy ##
                for each_val in range(decoder_output_vals.size(0)):
                    decoder_output_val = decoder_output_vals[each_val]
                    predicted_val = []
                    for i in decoder_output_val:
                        _, topi = i.topk(1)
                        pred_val = all_note[topi.item()]
                        predicted_val.append(pred_val)
                    expected_val = [all_note[i] for i in target_tensor_val[each_val]]
                
                    acc_all_val, acc_time_val = sequence_accuracy(predicted_val, expected_val)

                    total_acc_all_val.append(acc_all_val)
                    total_acc_time_val.append(acc_time_val)

                    
                    # If it's last epoch, save result in csv file #
                    if epoch == num_epochs-1:
                        new_row = pd.DataFrame([{"target":f"{expected}", "predict":f"{predicted}"}])
                        df = pd.concat([df, new_row])
                        df.reset_index(drop=True, inplace=True)
            
            if epoch%10 == 0:
                # wandb.log({"epoch": epoch, 'Confusion Matrix/epoch': wandb.Image(plot_confusion_matrix(cf_matrix_all, rhythm_to_id))})
                wandb.log({"epoch": epoch,'Loss/validation': np.average(total_loss_val)})
                wandb.log({"epoch": epoch,'AccuracyAll/validation': np.average(total_acc_all_val)})
                wandb.log({"epoch": epoch,'AccuracyTime/validation': np.average(total_acc_time_val)})

            print(f"Validate Set, Loss: {np.average(total_loss_val):.4}, Accuracy All: {np.average(total_acc_all_val):.4}, Accuracy Time: {np.average(total_acc_time_val):.4}")

            # if epoch%10 == 0:
            #     wandb.log({"epoch": epoch, 'Total.Loss/train': sum(total_loss), 'Total.Loss/validation': sum(total_loss_val)})
            #     wandb.log({"epoch": epoch, "Avg.Loss/train": np.average(total_loss), "Avg.Loss/validation": np.average(total_loss_val)})
            #     wandb.log({"epoch": epoch, "Avg.Accuracy/train": np.average(total_AvgAcc), "Avg.Accuracy/validation": np.average(total_AvgAcc_val)})
            #     wandb.log({"epoch": epoch, "EOA.Accuracy/train": np.average(total_EOA), "EOA.Accuracy/validation": np.average(total_EOA_val)})
            #     wandb.log({"epoch": epoch, "LA.Accuracy/train": np.average(total_LA), "LA.Accuracy/validation": np.average(total_LA_val)})


    ## Trainning Loop End Here ----------------------------------------- ##

    # if experiment_name is not None:
    #     print(df)
    #     df.to_csv(f"{LOG_PATH}/train_test_predict.csv", index=False)

    # result_table = pd.DataFrame({
    #     "Metric" : ["total loss - train", "avg loss - train", "avg acc - train", 
    #                 "total loss - validation", "avg loss - validation", "avg acc - validation"],
    #     "Value" : [round(sum(total_loss), 4), round(np.average(total_loss), 4), round(np.average(total_AvgAcc), 4),
    #                round(sum(total_loss_val), 4), round(np.average(total_loss_val), 4), round(np.average(total_AvgAcc_val), 4)]
    # })

    # wandb.log({"Result of Last Epoch": result_table})

    


### Use for Logging ----------------------------------------------------###

def plot_histrogram(name, label):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    all_label = [j for i in label for j in ast.literal_eval(i)]
    plt.hist(all_label, density = True, width=0.5)
    plt.xticks([i for i in rhythm_to_id.keys()])
    ax.set_axisbelow(True)
    ax.grid()
    return fig

def visualize_data(idx_train, idx_val, train_dir, val_dir, wandb):
    # train set
    train_label = pd.read_csv(train_dir+"/metadata.csv")
    train_label = train_label.loc[idx_train.tolist()]["rhythm_label"].to_list()

    # test set
    test_label = pd.read_csv(val_dir+"/metadata.csv")
    test_label = test_label.loc[idx_val.tolist()]["rhythm_label"].to_list()

    wandb.log({f'Histrogram/Train {train_dir}': plot_histrogram(f"{train_dir}", train_label),
               f'Histrogram/Test {val_dir}': plot_histrogram(f"{val_dir}", test_label)})

    dataset_detail = pd.DataFrame({
        "train" : [str(list(idx_train))],
        "validate" : [str(list(idx_val))]
    }, index=[0])

    wandb.log({"dataset_detail": dataset_detail})



if __name__ == "__main__":

    ### ------------ Variable / Parameter ----------------- ###
    force_start = "YES"

    CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
    SAMPLE_RATE = 22050
    n_fft = 2048
    hop_length = 1025
    n_mels = 256

    time_length = 100
    overlap = 50
    max_length = 60
    # batch_size = 10
    input_size = n_mels*time_length
    hidden_size = 128
    encoder_heads = 1
    decoder_heads = 1
    encoder_layers = 1
    decoder_layers = 1

    lr = 0.00001
    num_epochs = 2000

    rhythm_to_id = {"<SOS>":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "<PAD>": 6, "<EOS>":7}
    ID_TO_RHYTHM = {v: k for k, v in rhythm_to_id.items()}
    audio_dir = "dataset/dif_length_data"
    validation_dir = None
    experiment_name = "dif_length_conv2d_emb"
    model_type = "transformer"
    log_tags = [model_type, audio_dir]
    log_notes = "no flatten, emb with conv2d 1 layer"
    random_state = 0

    #------------------------------------------------------#

    # Default variable with condition #
    if validation_dir is None: 
        validation_dir = audio_dir 
        validate_size = 0.3
    if experiment_name is None: 
        experiment_name = "Temp"

    LOG_PATH = f'{str(CURRENT_PATH)}/log/{audio_dir}/{str(experiment_name)}{time.strftime("%Y%m%d-%H%M", time.localtime())}'
    directory_path = pathlib.Path(LOG_PATH)
    directory_path.mkdir(parents=True, exist_ok=True)

    MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels = n_mels
        )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device {DEVICE}")

    # Initialize WandB
    wandb.init(project="senior-project", config={
        "model": model_type,
        "dataset" : audio_dir,
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        'SAMPLE_RATE': SAMPLE_RATE, 'n_fft':n_fft, 'hop_length': hop_length, 'n_mels':n_mels,
        'time_length':time_length, 'overlap':overlap, 'max_length': max_length, 
        'input_size':input_size, 'hidden_size':hidden_size, 
        'encoder_heads':encoder_heads, 'encoder_layers':encoder_layers,
        'decoder_heads':decoder_heads, 'decoder_layers':decoder_layers
        # Add other hyperparameters you want to track
    }, dir=LOG_PATH, name=experiment_name, tags=log_tags, notes=log_notes)
    config = wandb.config


    train_sound = MusicDataset(audio_dir + "/metadata.csv",
                    audio_dir,
                    MEL_SPECTROGRAM,
                    SAMPLE_RATE,
                    time_length,
                    overlap,
                    rhythm_to_id,
                    max_length,
                    DEVICE,
                    wandb
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
                        wandb
                        )
        train_set = DataLoader(train_sound, batch_size=batch_size, collate_fn=custom_collate_fn(rhythm_to_id))
        validation_set = DataLoader(validate_sound, batch_size=batch_size, collate_fn=custom_collate_fn(rhythm_to_id))
        idx_train, idx_validation = np.arange(len(train_sound)), np.arange(len(validate_sound))
    else:
        train_set, validation_set, idx_train, idx_validation = train_test_split(train_sound, np.arange(len(train_sound)), test_size=validate_size, random_state=random_state, shuffle=True)

    print(f"There are {len(train_sound)} samples in the dataset.")
    signal_shape = set()
    for i in range(len(train_sound)): 
        signal, target_onehot, label = train_sound[i]; signal_shape.add(signal.shape)

    visualize_data(idx_train, idx_validation, audio_dir, validation_dir, wandb)

    model = TransformerConv2dModel(input_size, hidden_size, output_dict, max_length,
                           encoder_heads, encoder_layers, decoder_heads, decoder_layers).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    parameter_table = pd.DataFrame({
        "Parameter" : ['unique signal shape', 'time_length', 'overlap', 'batch_size', 'n_fft', 'hop_length', 'n_mels', 
                       'input_size', 'hidden_size', 'encoder_heads', 'encoder_layers', 'decoder_heads', 'decoder_layers',
                       'num_epochs', 'learning_rate'],
        "Value" : [str(signal_shape), str(time_length), str(overlap), str(batch_size), str(n_fft), str(hop_length), str(n_mels),
                   str(input_size), str(hidden_size), str(encoder_heads), str(encoder_layers), str(decoder_heads), str(decoder_layers),
                   str(num_epochs), str(lr)]
        })
    
    wandb.log({"parameter_table": parameter_table})


    print("Recheck Your Parameter")
    print(parameter_table)
    print("You want to continue?")
    force_start = str(input("YES|NO : ")) if force_start is None else "YES"
    if (force_start == "YES" ):
        print("OKiee, about to start...")

        start = time.time()

        train(model, num_epochs, train_set, criterion, model_optimizer, validation_set, max_length, wandb)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'output_dict': output_dict, 
            'rhythm_to_id':rhythm_to_id, 'ID_TO_RHYTHM':ID_TO_RHYTHM, 'audio_dir': audio_dir,
            'SAMPLE_RATE': SAMPLE_RATE, 'n_fft':n_fft, 'hop_length': hop_length, 'n_mels':n_mels,
            'time_length':time_length, 'overlap':overlap, 'max_length': max_length, 
            'input_size':input_size, 'hidden_size':hidden_size, 
            'encoder_heads':encoder_heads, 'encoder_layers':encoder_layers,
            'decoder_heads':decoder_heads, 'decoder_layers':decoder_layers
            }, f"{LOG_PATH}/model.pth")
        wandb.log_artifact(f"{LOG_PATH}/model.pth", type='model')
        print("Model trained and stored at model.pth")

        # print(Inference(audio_dir, LOG_PATH))
        # print(InferencePlot(LOG_PATH, rhythm_to_id, wandb))
        wandb.finish()
    else:
        shutil.rmtree(LOG_PATH)
        
