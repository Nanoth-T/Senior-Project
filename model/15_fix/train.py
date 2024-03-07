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


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def visualize_mel_spectrogram(tensor, title="Mel Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(tensor[0].numpy(), cmap='viridis', origin='lower', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel Frequency Bin')
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):

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
    cm = np.around(cm.astype('float') / row_sums[:, np.newaxis], decimals=4)
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def train(encoder, decoder, num_epochs, sound, criterion, 
          encoder_optimizer, decoder_optimizer, validate_set, device, writer):
    
    best_validation_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        encoder.train()
        decoder.train()

        # training process
        total_loss = []
        total_acc = []
        for batch in sound:
            signal_tensor, target_onehot, target_tensor = batch
            signal_tensor, target_onehot, target_tensor = signal_tensor.to(device), target_onehot.to(device), target_tensor.to(device)
            # mel_spectrogram = visualize_mel_spectrogram(signal_tensor, f'Mel Spectrogram - Epoch {epoch+1}')
            # writer.add_figure(f'Mel Spectrogram/{epoch * len(sound)}', mel_spectrogram, global_step=None, close=True, walltime=None)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # print(target_tensor)

            encoder_outputs, encoder_hidden = encoder(signal_tensor)
            # print(encoder_hidden[0].shape, encoder_hidden[1].shape)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_onehot=target_onehot)

            # print(decoder_outputs)
            # print(decoder_outputs.size())

            loss = criterion(
                    decoder_outputs.view(-1, 7),
                    target_tensor.view(-1)
                    )

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss.append(loss.item())

            acc = multiclass_accuracy(decoder_outputs.view(-1, 7), target_tensor.view(-1), 
                                      num_classes=len(rhythm_dict), ignore_index=-1).to(device)
            total_acc.append(acc.tolist())

        writer.add_scalar('Loss/train', sum(total_loss), epoch)
        writer.add_scalar('Avg.Loss/train', np.average(total_loss), epoch)   
        writer.add_scalar('Accuracy/train', np.average(total_acc), epoch)

        print(f"{timeSince(start)} (Epoch {epoch+1}/{num_epochs}), Loss: {sum(total_loss)}, Accuracy: {np.average(total_acc)}")

    
        # validation compute

        encoder.eval()
        decoder.eval()


        with torch.no_grad():
            confmat = ConfusionMatrix(task="multiclass", num_classes=len(rhythm_dict), ignore_index=-1).to(device)
            total_loss_val = []
            total_acc_val = []
            cf_matrix_all = np.zeros((len(rhythm_dict),len(rhythm_dict)))
            for b in validate_set:
                signal_val, target_onehot_val, target_val = b
                signal_val, target_onehot_val, target_val = signal_val.to(device), target_onehot_val.to(device), target_val.to(device)
                encoder_outputs_val, encoder_hidden_val = encoder(signal_val)
                decoder_outputs_val, _, _ = decoder(encoder_outputs_val, encoder_hidden_val, target_onehot=target_onehot_val)
                
                loss_val = criterion(
                    decoder_outputs_val.view(-1, 7),
                    target_val.view(-1)
                    )
                total_loss_val.append(loss_val.item())
                acc_val = multiclass_accuracy(decoder_outputs_val.view(-1, 7), target_val.view(-1), 
                                              num_classes=len(rhythm_dict), ignore_index=-1).to(device)
                total_acc_val.append(acc_val.tolist())

                cf_matrix = confmat(decoder_outputs_val.view(-1, 7), target_val.view(-1)).cpu().numpy()
                cf_matrix_all += cf_matrix
        
            if (epoch+1)%5 == 0:
                writer.add_figure(f'Confusion matrix', plot_confusion_matrix(cf_matrix_all, rhythm_dict), epoch)


            print(f"Validation Set, Loss: {sum(total_loss_val)}, Accuracy: {np.average(total_acc_val)}")

            if sum(total_loss_val) < best_validation_loss:
                best_validation_loss = sum(total_loss_val)
                torch.save(encoder.state_dict(), f"{str(current_path)}/model/enmodel_{str(AUDIO_DIR)[str(AUDIO_DIR).find('/')+1::]}.pth")
                torch.save(decoder.state_dict(), f"{str(current_path)}/model/demodel_{str(AUDIO_DIR)[str(AUDIO_DIR).find('/')+1::]}.pth")

                print(f"{'-'*15}save model.{'-'*15}")
            
            writer.add_scalar('Loss/validation', sum(total_loss_val), epoch)
            writer.add_scalar('Avg.Loss/validation', np.average(total_loss_val), epoch)      
            writer.add_scalar('Accuracy/validation', np.average(total_acc_val), epoch)



if __name__ == "__main__":

    current_path = pathlib.Path(__file__).parent.resolve()
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    AUDIO_DIR = "dataset/train_1000"
    ANNOTATIONS_FILE = "/metadata.csv"
    VALIDATE_DIR = "dataset/test_150" # might 80-20 with large training samples
    writer = SummaryWriter(f'{str(current_path)}/log/{str(AUDIO_DIR)}/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}')
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

    # print(f"There are {len(sound)} samples in the dataset.")
    # for i in range(len(sound)):
    #     signal,target_onehot, label = sound[i]
    #     print(signal.shape)

    train_data_loader = create_data_loader(train_sound, batch_size)
    validate_data_loader = create_data_loader(validate_sound, batch_size)

    # print(data_loader)

    input_size = n_mels*time_length
    hidden_size = 128
    output_size = len(rhythm_dict)
    max_length = 60


    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_size, max_length).to(device)

    lr = 0.005

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    num_epochs = 1000

    loss_acc = {
            "loss-acc": {
                "total loss": ["Multiline", ["Loss/train", "Loss/validation"]],
                "avg loss": ["Multiline", ["Avg.Loss/train", "Avg.Loss/validation"]],
                "accuracy": ["Multiline", ["Accuracy/train", "Accuracy/validation"]],
            },
        }

    writer.add_custom_scalars(loss_acc)

    table = f"""
        | Parameter |Value|
        |----------|-----------|
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


    train(encoder, decoder, num_epochs, train_data_loader, criterion, encoder_optimizer, decoder_optimizer, validate_data_loader, device, writer)
    torch.save(encoder.state_dict(), f"{str(current_path)}/model/last_enmodel_{str(AUDIO_DIR)[str(AUDIO_DIR).find('/')+1::]}.pth")
    torch.save(decoder.state_dict(), f"{str(current_path)}/model/last_demodel_{str(AUDIO_DIR)[str(AUDIO_DIR).find('/')+1::]}.pth")
    print("Model trained and stored at model.pth")

    
