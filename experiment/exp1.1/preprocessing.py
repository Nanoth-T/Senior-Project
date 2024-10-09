import torchaudio
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
import pathlib
import ast
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import ctypes

class MusicDataset(Dataset):
    def __init__(self,
               annotations_file,
               audio_dir,
               transformation,
               target_sample_rate,
               time_length,
               rhythm_dict,
               max_length,
               device,
               writer=None
               ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.time_length = time_length
        self.rhythm_dict = rhythm_dict
        self.max_length = max_length
        self.writer = writer


    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 1]
        label = ast.literal_eval(label)
        return label
    
    def __len__(self):
        return len(self.annotations)
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # (2, 1000) --> 2
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _targetTensor(self, label):
        rt_indexes = [self.rhythm_dict[rt] for rt in label]
        rt_indexes.append(self.rhythm_dict["EOS"]) # EOS
        # rt_indexes.insert(0, self.rhythm_dict["SOS"])
        # while len(rt_indexes) < self.max_length:
        #     rt_indexes.append(-1)
        return torch.LongTensor(rt_indexes)
    
    def _target_onehot(self, label):
        target_onehot_tensor = torch.zeros(len(label)+1, 1, len(self.rhythm_dict))
        for i in range(len(label)+1):
            if i == 0:
                target_onehot_tensor[i][0][self.rhythm_dict["SOS"]] = 1
                continue
            rt = label[i-1]
            target_onehot_tensor[i][0][self.rhythm_dict[rt]] = 1
        # input_tensor[-1][0][self.rhythm_dict["EOS"]] = 1
        return target_onehot_tensor
    
    def plot_spectrogram(self, specgram, title, subtitle=0, ylabel="freq_bin", ax=None):
        writer = self.writer
        # print(specgram.shape)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if subtitle is not None:
            ax.set_title(f"Mel Spectrogram of sound {title}-{subtitle}")
        else:
            ax.set_title(f"Mel Spectrogram of sound {title}")
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram.cpu()), origin="lower", aspect="auto", interpolation="nearest")
        if writer is not None:
            writer.add_figure(f'Mel Spectrogram {title}', fig, subtitle)
        # plt.savefig(f"{title}.png")
        # plt.show()
        # return fig

    def split_melspectrogram(self, signal):
        seq_len = math.ceil(signal.size(2) / self.time_length)
        total_size = signal.size(2)
        split_size = self.time_length
        remainder = total_size % split_size
        # print(total_size, seq_len, remainder)
        split_tensors = torch.split(signal, split_size, dim=2)

        # If there is a remainder, pad the last tensor
        if remainder > 0:
            last_tensor = split_tensors[-1]
            padding_size = split_size - last_tensor.size(2)
            last_tensor = torch.nn.functional.pad(last_tensor, (0, padding_size))
            split_tensors = split_tensors[:-1] + (last_tensor,)


        return split_tensors

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label_tensor = self._targetTensor(label) 
        target_onehot = self._target_onehot(label)

        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)

        mel_signal = self.transformation(signal)
        self.plot_spectrogram(mel_signal[0], f"{index}")
        # signal_tensor = mel_signal.reshape(1, mel_signal.shape[2], mel_signal.shape[1])
        # self.plot_spectrogram(signal_tensor[0])
        sequences = self.split_melspectrogram(mel_signal)
        for s in range(len(sequences)):
            self.plot_spectrogram(sequences[s][0], f"{index}", f"{s+1}")
        sequences_tensor = torch.stack(sequences)
        sequences_tensor = torch.flatten(sequences_tensor, start_dim=1, end_dim=-1)


        return sequences_tensor, target_onehot, label_tensor

def create_data_loader(data, batch_size):

    signal_list, onehot_list, label_list = [], [], []
    num_batch = math.ceil(len(data) / batch_size)
    for i in range(len(data)):
        signal, target_onehot, label = data[i]
        signal_list.append(signal)
        onehot_list.append(target_onehot)
        label_list.append(label)


    train_data_loader = []
    start = 0
    for batch in range(1, num_batch+1):
        # print(onehot_list[start:batch_size*batch])
        signal_batch = pad_sequence(signal_list[start:batch_size*batch], padding_value=0,batch_first=True)
        # print(signal_batch.shape)
        onehot_batch = pad_sequence(onehot_list[start:batch_size*batch], padding_value=0,batch_first=True)
        # print(onehot_batch.shape)
        # for i in range(onehot_batch.size(0)):
        #     print(onehot_batch[i,:])
        label_batch = pad_sequence(label_list[start:batch_size*batch], padding_value=-1,batch_first=True)
        # print(label_batch.shape)
        each_batch = (signal_batch, onehot_batch, label_batch)
        train_data_loader.append(each_batch)

        # for sp in range(len(signal_batch)):
        #     # print(signal_batch[sp])
        #     plot_spectrogram(signal_batch[sp], sp)

        start = batch_size*batch
        

    return train_data_loader

def plot_spectrogram(specgram, title, subtitle=0, ylabel="freq_bin", ax=None):
        # print(specgram.shape)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if subtitle is not None:
            ax.set_title(f"Mel Spectrogram of sound {title}-{subtitle}")
        else:
            ax.set_title(f"Mel Spectrogram of sound {title}")
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram.cpu()), origin="lower", aspect="auto", interpolation="nearest")
        # plt.savefig(f"{title}-{subtitle}.png")



if __name__ == "__main__":

    current_path = pathlib.Path(__file__).parent.resolve()
    rhythm_dict = {"SOS":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "rest_quarter": 6, "EOS":7}
    AUDIO_DIR = "dataset/train_100"
    ANNOTATIONS_FILE = AUDIO_DIR + "/metadata.csv"
    SAMPLE_RATE = 22050
    time_length = 400
    max_length = 60
    batch_size = 10

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 2048,
        # hop_length = 1025,
        n_mels = 256
        )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    sound = MusicDataset(ANNOTATIONS_FILE,
                    AUDIO_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    rhythm_dict,
                    max_length,
                    device
                    )

    print(f"There are {len(sound)} samples in the dataset.")
    for i in range(len(sound)):
        signal,target_onehot, label = sound[i]
        # print(signal)

    train, test, idx_train, idx_test = train_test_split(sound, np.arange(len(sound)), test_size=0.3, shuffle=True)
    print(train)
    print(idx_train)
    print("-----------------------")
    print(test)
    print(idx_test)
    train_data_loader = create_data_loader(sound, batch_size)

    print(len(train_data_loader))

    for batch in train_data_loader:
        signals, target_onehots, target_labels = batch
        print("Signals Shape:", signals.shape)
        print("Onehots Shape:", target_onehots.shape)
        print("Labels Shape:", target_labels.shape)



            
