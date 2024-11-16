### Import Fuctions and Class -----------------###
from preprocessing import MusicDataset_Chunk, MusicDataset_Unchunk, custom_collate_fn
# from inference import Inference
# from inference_plot import InferencePlot

### Import Library -----------------------------###
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
# from torchmetrics import ConfusionMatrix

import wandb

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pathlib
import ast
import shutil

import pandas as pd
# from tabulate import tabulate
import numpy as np
import itertools
import time
import math
import matplotlib.pyplot as plt
# from morefunc import EarlyStopping

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
    RANGE_TIME_SHIFT = list(range(256, 756))  # 256-755
    RANGE_VEL = list(range(756, 884))  # 756-884

    # print("pred:", pred)
    # print("target:", target)
    if type(pred) != list:
        pred = ast.literal_eval(pred)
    if type(target) != list:
        target = ast.literal_eval(target)

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

    # early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    ### Trainning loop ------------------------ #

    for epoch in range(1, num_epochs+1):
        print(f"Epoch [{epoch}/{num_epochs}]")

        ### Create Batch ###
        train_set_shuffle = shuffle(train_set)
        validation_set_shuffle = shuffle(validation_set)

        batch_train = DataLoader(train_set_shuffle, batch_size=batch_size, collate_fn=custom_collate_fn(output_dict))
        batch_val = DataLoader(validation_set_shuffle, batch_size=1, collate_fn=custom_collate_fn(output_dict))

        ### Mode train ###
        model.train()

        

        total_loss = []
        total_acc_all = []
        total_acc_time = []

        for batch in batch_train:
            signal_tensor, input_tensor, target_tensor, _ = batch
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

        # if epoch%5 == 0:
        wandb.log({"epoch": epoch, 'Loss/train': np.average(total_loss)})
        wandb.log({"epoch": epoch, 'AccuracyAll/train': np.average(total_acc_all)})
        wandb.log({"epoch": epoch, 'AccuracyTime/train': np.average(total_acc_time)})

        print(f"{timeSince(start)} (Epoch {epoch}/{num_epochs}), Loss: {np.average(total_loss):.4}, Accuracy All: {np.average(total_acc_all):.4}, , Accuracy Time: {np.average(total_acc_time):.4}")

    
        ### Mode validate ###
        if epoch%25 == 0:
            model.eval()

            with torch.no_grad():
                # confmat = ConfusionMatrix(task="multiclass", num_classes=len(rhythm_to_id), ignore_index=-1).to(DEVICE)
                total_loss_val = []
                total_acc_all_val=[]
                total_acc_time_val=[]
                # cf_matrix_all = np.zeros((len(rhythm_to_id),len(rhythm_to_id)))

                for batch in batch_val:
                    signal_tensor_val, input_tensor_val, target_tensor_val, _ = batch
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
                
                

                if epoch%5 == 0:
                    # wandb.log({"epoch": epoch, 'Confusion Matrix/epoch': wandb.Image(plot_confusion_matrix(cf_matrix_all, rhythm_to_id))})
                    wandb.log({"epoch": epoch,'Loss/validation': np.average(total_loss_val)})
                    wandb.log({"epoch": epoch,'AccuracyAll/validation': np.average(total_acc_all_val)})
                    wandb.log({"epoch": epoch,'AccuracyTime/validation': np.average(total_acc_time_val)})

                print(f"Validate Set, Loss: {np.average(total_loss_val):.4}, Accuracy All: {np.average(total_acc_all_val):.4}, Accuracy Time: {np.average(total_acc_time_val):.4}")

            # early_stopping(np.average(total_loss_val))
            # if early_stopping.early_stop:
            #     print("Early stopping triggered. Stopping training.")
            #     break
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

# def plot_histrogram(name, label):
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     all_label = [j for i in label for j in ast.literal_eval(i)]
#     plt.hist(all_label, density = True, width=0.5)
#     plt.xticks([i for i in rhythm_to_id.keys()])
#     ax.set_axisbelow(True)
#     ax.grid()
#     return fig

# def visualize_data(idx_train, idx_val, train_dir, val_dir, wandb):
#     # train set
#     train_label = pd.read_csv(train_dir+"/metadata.csv")
#     train_label = train_label.loc[idx_train.tolist()]["rhythm_label"].to_list()

#     # test set
#     test_label = pd.read_csv(val_dir+"/metadata.csv")
#     test_label = test_label.loc[idx_val.tolist()]["rhythm_label"].to_list()

#     wandb.log({f'Histrogram/Train {train_dir}': plot_histrogram(f"{train_dir}", train_label),
#                f'Histrogram/Test {val_dir}': plot_histrogram(f"{val_dir}", test_label)})

#     dataset_detail = pd.DataFrame({
#         "train" : [str(list(idx_train))],
#         "validate" : [str(list(idx_val))]
#     }, index=[0])

#     wandb.log({"dataset_detail": dataset_detail})

