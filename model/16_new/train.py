import torch
import torch.nn as nn
from model import EncoderRNN, DecoderRNN
from preprocessing import MusicDataset, create_data_loader
import torchaudio
import time
import math
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics import ConfusionMatrix
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
import ast


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def plot_confusion_matrix(cm, class_names):

    raw_cm = cm
    figure = plt.figure(figsize=(8, 8))
    
    # dark mode
    # df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index = [i for i in class_names],
    #                  columns = [i for i in class_names])
    # plt.figure(figsize = (12,7))
    # figure = sns.heatmap(df_cm, annot=True).get_figure()
    
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



def train(encoder, decoder, num_epochs, sound, criterion, 
          encoder_optimizer, decoder_optimizer, validate_set, max_length, writer):
    
    best_validation_loss = float('inf')
    csv = {
        'target':[None],
        'predict':[None]
    }
    df = pd.DataFrame(csv)
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        encoder.train()
        decoder.train()

        # training process
        total_loss = []
        # total_acc = []
        all_acc_exactly = []
        all_acc_len = []
        all_acc_in = []
        all_acc_avg = []
        for batch in sound:
            signal_tensor, target_onehot, target_tensor = batch
            signal_tensor, target_onehot, target_tensor = signal_tensor.to(device), target_onehot.to(device), target_tensor.to(device)
            # mel_spectrogram = visualize_mel_spectrogram(signal_tensor, f'Mel Spectrogram - Epoch {epoch+1}')
            # writer.add_figure(f'Mel Spectrogram/{epoch * len(sound)}', mel_spectrogram, global_step=None, close=True, walltime=None)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # print(target_onehot)

            encoder_outputs, encoder_hidden = encoder(signal_tensor)
            # print(encoder_hidden[0].shape, encoder_hidden[1].shape)
            decoder_outputs, _ = decoder(encoder_hidden, target_onehot=target_onehot)


            loss = criterion(
                    decoder_outputs.view(-1, output_size),
                    target_tensor.view(-1)
                    )

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss.append(loss.item())

            # print(decoder_outputs)
            # print(decoder_outputs.size())
            for b in range(decoder_outputs.size(0)):
                # print(decoder_outputs.size(0), target_tensor.size(0))

                decoder_output = decoder_outputs[b]
                # print(decoder_output.size())
                predicted = []
                for elm in decoder_output:
                    topv, topi = elm.topk(1)
                    if topi.item() == rhythm_dict["EOS"]:
                        predicted.append(rhythm_dict_swap[topi.item()])
                        break
                    predicted.append(rhythm_dict_swap[topi.item()])
                expected = list(map(rhythm_dict_swap.get, target_tensor[b].tolist()))
                expected = expected[:expected.index("EOS")+1]

                if epoch == num_epochs-1:
                    new_row = pd.DataFrame([{"target":f"{expected}", "predict":f"{predicted}"}])
                    df = pd.concat([df, new_row])
                    df.reset_index(drop=True, inplace=True)

                acc_exactly, acc_len, acc_in, acc_avg = sequence_accuracy(predicted, expected)

                all_acc_exactly.append(acc_exactly)
                all_acc_len.append(acc_len)
                all_acc_in.append(acc_in)
                all_acc_avg.append(acc_avg)

            # acc = multiclass_accuracy(decoder_outputs.view(-1, output_size), target_tensor.view(-1), 
            #                           num_classes=len(rhythm_dict), ignore_index=-1).to(device)
            
            # total_acc.append(acc.tolist())

        writer.add_scalar('Loss/train', sum(total_loss), epoch)
        writer.add_scalar('Avg.Loss/train', np.average(total_loss), epoch)   
        writer.add_scalar('Avg.Accuracy/train', np.average(all_acc_avg), epoch)
        writer.add_scalar('Exac.Accuracy/train', np.average(all_acc_exactly), epoch)
        writer.add_scalar('Len.Accuracy/train', np.average(all_acc_len), epoch)
        writer.add_scalar('In.Accuracy/train', np.average(all_acc_in), epoch)



        print(f"{timeSince(start)} (Epoch {epoch+1}/{num_epochs}), Loss: {sum(total_loss)}, Accuracy: {np.average(all_acc_avg)}")

    
        # validation compute - no batch

        encoder.eval()
        decoder.eval()


        with torch.no_grad():
            confmat = ConfusionMatrix(task="multiclass", num_classes=len(rhythm_dict), ignore_index=-1).to(device)
            total_loss_val = []
            all_acc_exactly_val = []
            all_acc_len_val = []
            all_acc_in_val = []
            all_acc_avg_val = []
            cf_matrix_all = np.zeros((len(rhythm_dict),len(rhythm_dict)))
            for batch_val in validate_set:
                signal_val, target_onehot_val, target_val = batch_val
                signal_val, target_onehot_val, target_val = signal_val.to(device), target_onehot_val.to(device), target_val.to(device)
                encoder_outputs_val, encoder_hidden_val = encoder(signal_val)
                decoder_outputs_val, _ = decoder(encoder_hidden_val, max_loop=max_length)
                # print(target_val)

                pad_target_val = torch.nn.functional.pad(target_val, (0, decoder_outputs_val.size(1)-target_val.size(1)), value=-1)


                cf_matrix = confmat(decoder_outputs_val.view(-1, output_size), pad_target_val.view(-1)).cpu().numpy()
                cf_matrix_all += cf_matrix

                loss_val = criterion(
                    decoder_outputs_val.view(-1, output_size),
                    pad_target_val.view(-1)
                    )
                
                total_loss_val.append(loss_val.item())

                for b_val in range(decoder_outputs_val.size(0)):

                    decoder_output_val = decoder_outputs_val[b_val]
                    # print(decoder_output.size())
                    predicted_val = []
                    for elm in decoder_output_val:
                        topv, topi = elm.topk(1)
                        if topi.item() == rhythm_dict["EOS"]:
                            predicted_val.append(rhythm_dict_swap[topi.item()])
                            break
                        predicted_val.append(rhythm_dict_swap[topi.item()])
                    expected_val = list(map(rhythm_dict_swap.get, target_val[b_val].tolist()))
                    expected_val = expected_val[:expected_val.index("EOS")+1]

                    if epoch == num_epochs-1:
                        new_row = pd.DataFrame([{"target":f"{expected_val}", "predict":f"{predicted_val}"}])
                        df = pd.concat([df, new_row])
                        df.reset_index(drop=True, inplace=True)

                    acc_exactly_val, acc_len_val, acc_in_val, acc_avg_val = sequence_accuracy(predicted_val, expected_val)
                    all_acc_exactly_val.append(acc_exactly_val)
                    all_acc_len_val.append(acc_len_val)
                    all_acc_in_val.append(acc_in_val)
                    all_acc_avg_val.append(acc_avg_val)




                


                # loss_val = criterion(
                #     decoder_outputs_val.view(-1, output_size),
                #     target_val.view(-1)
                #     )

                

                # acc_val = multiclass_accuracy(decoder_outputs_val.view(-1, output_size), 
                #                               target_val.view(-1), 
                #                               num_classes=len(rhythm_dict), ignore_index=-1).to(device)
                # total_acc_val.append(acc_val.tolist())

                
            if (epoch+1)%5 == 0:
                writer.add_figure(f'Confusion matrix', plot_confusion_matrix(cf_matrix_all, rhythm_dict), epoch)


            print(f"Test Set, Loss: {sum(total_loss_val)}, Accuracy: {np.average(all_acc_avg_val)}")

            if sum(total_loss_val) < best_validation_loss:
                best_validation_loss = sum(total_loss_val)
                if experiment is None:
                    torch.save(encoder.state_dict(), f"{str(current_path)}/model/enmodel_{str(AUDIO_DIR)[str(AUDIO_DIR).find('/')+1::]}.pth")
                    torch.save(decoder.state_dict(), f"{str(current_path)}/model/demodel_{str(AUDIO_DIR)[str(AUDIO_DIR).find('/')+1::]}.pth")
                else:
                    torch.save(encoder.state_dict(), f"{str(current_path)}/exp_log/{str(experiment)}/enmodel.pth")
                    torch.save(decoder.state_dict(), f"{str(current_path)}/exp_log/{str(experiment)}/demodel.pth")

                print(f"{'-'*15}save model.{'-'*15}")
            
            writer.add_scalar('Loss/validation', sum(total_loss_val), epoch)
            writer.add_scalar('Avg.Loss/validation', np.average(total_loss_val), epoch)      
            writer.add_scalar('Avg.Accuracy/validation', np.average(all_acc_avg_val), epoch)
            writer.add_scalar('Exac.Accuracy/validation', np.average(all_acc_exactly_val), epoch)
            writer.add_scalar('Len.Accuracy/validation', np.average(all_acc_len_val), epoch)
            writer.add_scalar('In.Accuracy/validation', np.average(all_acc_in_val), epoch)

    if experiment is not None:
        print(df)
        df.to_csv(f"{str(current_path)}/exp_log/{str(experiment)}/train_test_predict.csv", index=False) 

    result = f"""
        | Metric | Value |
        |----------|-----------|
        | total loss train | {sum(total_loss):.4f} |
        | avg loss train | {np.average(total_loss):.4f} |
        | avg acc train | {np.average(all_acc_avg):.4f} |
        | total loss test | {sum(total_loss_val):.4f} |
        | avg loss test | {np.average(total_loss_val):.4f} |
        | avg acc test | {np.average(all_acc_avg_val):.4f} |
    """
    result = '\n'.join(l.strip() for l in result.splitlines())
    writer.add_text("Last Epoch", result, 0)


    print("end")



def plot_histrogram(name, label):
    fig = plt.figure(figsize=(8, 8))
    all_label = [j for i in label for j in ast.literal_eval(i)]
    plt.hist(all_label, density = True)
    plt.xticks([i for i in rhythm_dict.keys()])
    return fig

def visualize_data(idx_train, idx_test, train_dir, test_dir, writer):
    # train set
    train_label = pd.read_csv(train_dir+"/metadata.csv")
    train_label = train_label.loc[idx_train.tolist()]["rhythm_label"].to_list()
    writer.add_figure(f'Histrogram Train {train_dir}', plot_histrogram(f"{train_dir}", train_label), 0)

    # test set
    test_label = pd.read_csv(test_dir+"/metadata.csv")
    test_label = test_label.loc[idx_test.tolist()]["rhythm_label"].to_list()
    writer.add_figure(f'Histrogram Test {test_dir}', plot_histrogram(f"{test_dir}", test_label), 0)

    dataset_detail = f"""
        | Name | Value |
        |--------------------|-------------------------------------------------------------|
        | train_dir | {train_dir} |
        | idx_train | {idx_train} |
        | test_dir | {test_dir} |
        | idx_test | {idx_test} |
    """


    dataset_detail = '\n'.join(l.strip() for l in dataset_detail.splitlines())
    writer.add_text("dataset_detail", dataset_detail, 0)



if __name__ == "__main__":

    current_path = pathlib.Path(__file__).parent.resolve()
    rhythm_dict = {"SOS":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "rest_quarter": 6, "EOS":7}
    # rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    AUDIO_DIR = "dataset/data4note"
    ANNOTATIONS_FILE = "/metadata.csv"
    VALIDATE_DIR = "dataset/data4note"

    experiment = f"{AUDIO_DIR}3rd_timelength200"
    # experiment = None
    if experiment is None:
        writer = SummaryWriter(f'{str(current_path)}/log/{str(AUDIO_DIR)}/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}')
    else:
        writer = SummaryWriter(f'{str(current_path)}/exp_log/{str(experiment)}/log')

    SAMPLE_RATE = 22050
    time_length = 200
    max_length = 60
    batch_size = 10
    n_mels = 256
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


    train_sound = MusicDataset(AUDIO_DIR + ANNOTATIONS_FILE,
                    AUDIO_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    rhythm_dict,
                    max_length,
                    device,
                    writer
                    )
    if VALIDATE_DIR != AUDIO_DIR:
        validate_sound = MusicDataset(VALIDATE_DIR+ANNOTATIONS_FILE,
                        VALIDATE_DIR,
                        mel_spectrogram,
                        SAMPLE_RATE,
                        time_length,
                        rhythm_dict,
                        max_length,
                        device,
                        writer
                        )
        train_data_loader = create_data_loader(train_sound, batch_size)
        validate_data_loader = create_data_loader(validate_sound, batch_size)
        idx_train, idx_test = np.arange(len(train_sound)), np.arange(len(validate_sound))

    else:
        train_set, test_set, idx_train, idx_test = train_test_split(train_sound, np.arange(len(train_sound)), test_size=0.3, shuffle=True)
        train_data_loader = create_data_loader(train_set, batch_size)
        validate_data_loader = create_data_loader(test_set, batch_size)


    # print(f"There are {len(sound)} samples in the dataset.")
    signal_shape = set()
    for i in range(len(train_sound)):
        signal,target_onehot, label = train_sound[i]
        signal_shape.add(signal.shape)

    


    visualize_data(idx_train, idx_test, AUDIO_DIR, VALIDATE_DIR, writer)
    # print(data_loader)

    input_size = n_mels*time_length
    hidden_size = 128
    output_dict = rhythm_dict
    output_size = len(rhythm_dict)
    max_length = 60


    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_dict, max_length).to(device)

    lr = 0.0005

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    num_epochs = 500

    loss_acc = {
            "loss-acc": {
                "total loss": ["Multiline", ["Loss/train", "Loss/validation"]],
                "avg loss": ["Multiline", ["Avg.Loss/train", "Avg.Loss/validation"]],
                "avg accuracy": ["Multiline", ["Avg.Accuracy/train", "Avg.Accuracy/validation"]],
                "acc exactly": ["Multiline", ["Exac.Accuracy/train", "Exac.Accuracy/validation"]],
                "len accuracy": ["Multiline", ["Len.Accuracy/train", "Len.Accuracy/validation"]],
                "in accuracy": ["Multiline", ["In.Accuracy/train", "In.Accuracy/validation"]],
            },
        }

    writer.add_custom_scalars(loss_acc)

    table = f"""
        | Parameter |Value|
        |----------|-----------|
        | signal shape | {signal_shape} |
        | train | {AUDIO_DIR} |
        | test | {VALIDATE_DIR} |
        | time_length | {time_length} |
        | batch_size | {batch_size} |
        | n_mels | {n_mels} |
        | hidden_size | {hidden_size} |
        | num_epochs | {num_epochs} |
        | learning_rate | {lr} |
    """
    table = '\n'.join(l.strip() for l in table.splitlines())
    writer.add_text("table", table, 0)



    start = time.time()


    train(encoder, decoder, num_epochs, train_data_loader, criterion, encoder_optimizer, decoder_optimizer, validate_data_loader, max_length, writer)
    if experiment is None:
        torch.save(encoder.state_dict(), f"{str(current_path)}/model/last_enmodel_{str(AUDIO_DIR)[str(AUDIO_DIR).find('/')+1::]}.pth")
        torch.save(decoder.state_dict(), f"{str(current_path)}/model/last_demodel_{str(AUDIO_DIR)[str(AUDIO_DIR).find('/')+1::]}.pth")
    else:
        torch.save(encoder.state_dict(), f"{str(current_path)}/exp_log/{str(experiment)}/last_enmodel.pth")
        torch.save(decoder.state_dict(), f"{str(current_path)}/exp_log/{str(experiment)}/last_demodel.pth")
    print("Model trained and stored at model.pth")

    
