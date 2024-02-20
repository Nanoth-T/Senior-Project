import torch
import torch.nn as nn
from model import EncoderRNN, DecoderRNN
from preprocessing import MusicDataset, create_data_loader
from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
import math
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pathlib
from torchmetrics.functional.classification import multiclass_accuracy


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

def train(encoder, decoder, num_epochs, sound, criterion, 
          encoder_optimizer, decoder_optimizer, validate_set, writer):
    
    best_validation_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        total_loss = 0
        total_acc = 0
        for batch in sound:
            signal_tensor, target_onehot, target_tensor = batch

            # mel_spectrogram = visualize_mel_spectrogram(signal_tensor, f'Mel Spectrogram - Epoch {epoch+1}')
            # writer.add_figure(f'Mel Spectrogram/{epoch * len(sound)}', mel_spectrogram, global_step=None, close=True, walltime=None)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # print(target_tensor)

            encoder_outputs, encoder_hidden = encoder(signal_tensor)
            # print(encoder_hidden[0].shape, encoder_hidden[1].shape)
            decoder_outputs, _ = decoder(encoder_hidden, target_onehot=target_onehot)

            # print(decoder_outputs)
            # print(decoder_outputs.size())

            loss = criterion(
                    decoder_outputs.view(-1, 7),
                    target_tensor.view(-1)
                    )

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()



            total_loss += loss.item()

            # for name, param in model.named_parameters():
            #     writer.add_histogram(name + '/gradients', param.grad, epoch * len(sound))

            writer.add_scalar('Loss/train', loss.item(), epoch * len(sound))

            # for name, param in model.named_parameters():
            #     writer.add_histogram(name, param.data, epoch * len(sound))
            
            # fixed += 1

            acc = multiclass_accuracy(decoder_outputs.view(-1, 7), target_tensor.view(-1), num_classes=len(rhythm_dict), ignore_index=-1)
            total_acc += acc
            writer.add_scalar('Accuracy/train', acc, epoch * len(sound))

        print(f"{timeSince(start)} (Epoch {epoch+1}/{num_epochs}), Loss: {total_loss}, Accuracy: {total_acc}")
        # torch.save(encoder.state_dict(), f"{str(current_path)}/en_checkpoint.pth")
        # torch.save(decoder.state_dict(), f"{str(current_path)}/de_checkpoint.pth")
    
        # validation compute
        # encoder.load_state_dict(torch.load(f"{str(current_path)}/en_checkpoint.pth"))
        # decoder.load_state_dict(torch.load(f"{str(current_path)}/de_checkpoint.pth"))
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            total_loss_val = 0
            total_acc_val = 0
            for b in validate_set:
                signal_val, target_onehot_val, target_val = b
                encoder_outputs_val, encoder_hidden_val = encoder(signal_val)
                decoder_outputs_val, _ = decoder(encoder_hidden_val, target_onehot=target_onehot_val)
                
                loss_val = criterion(
                    decoder_outputs_val.view(-1, 7),
                    target_val.view(-1)
                    )
                total_loss_val += loss_val.item()
                acc_val = multiclass_accuracy(decoder_outputs_val.view(-1, 7), 
                                              target_val.view(-1), 
                                              num_classes=len(rhythm_dict), ignore_index=-1)
                total_acc_val += acc_val


        print(f"Validation Set, Loss: {total_loss_val}, Accuracy: {total_acc_val}")

        if total_loss_val < best_validation_loss:
            best_validation_loss = total_loss_val
            torch.save(encoder.state_dict(), f"{str(current_path)}/enmodel.pth")
            torch.save(decoder.state_dict(), f"{str(current_path)}/demodel.pth")

            print(f"{'-'*15}save model.{'-'*15}")
                




if __name__ == "__main__":

    current_path = pathlib.Path(__file__).parent.resolve()
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    rhythm_dict_swap = {v: k for k, v in rhythm_dict.items()}
    AUDIO_DIR = "dataset/01/11"
    ANNOTATIONS_FILE = "/metadata.csv"
    VALIDATE_DIR = "dataset/01/09" # might 80-20 with large training samples
    writer = SummaryWriter(f'{str(current_path)}/log/{str(AUDIO_DIR)}/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}')
    SAMPLE_RATE = 22050
    time_length = 100
    max_length = 60
    batch_size = 10
    n_mels = 256

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 2048,
        # hop_length = 1025,
        n_mels = n_mels
        )


    train_sound = MusicDataset(AUDIO_DIR + ANNOTATIONS_FILE,
                    AUDIO_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    rhythm_dict,
                    max_length
                    )
    
    validate_sound = MusicDataset(VALIDATE_DIR+ANNOTATIONS_FILE,
                    VALIDATE_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    rhythm_dict,
                    max_length
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


    encoder = EncoderRNN(input_size, hidden_size)
    decoder = DecoderRNN(hidden_size, output_size, max_length)

    lr = 0.005

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    num_epochs = 100

    writer.add_text("time_length", f"time_length is {time_length}", 0)
    writer.add_text("batch_size", f"batch_size is {batch_size}", 0)
    writer.add_text("n_mels", f"n_mels is {n_mels}", 0)
    writer.add_text("hidden_size", f"hidden_size is {hidden_size}", 0)
    writer.add_text("num_epochs", f"num_epochs is {num_epochs}", 0)
    writer.add_text("learning_rate", f"learning_rate is {lr}", 0)



    start = time.time()


    train(encoder, decoder, num_epochs, train_data_loader, criterion, encoder_optimizer, decoder_optimizer, validate_data_loader, writer)
    # torch.save(encoder.state_dict(), f"{str(current_path)}/enmodel.pth")
    # torch.save(decoder.state_dict(), f"{str(current_path)}/demodel.pth")
    print("Model trained and stored at model.pth")

    
