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


class MusicDataset(Dataset):
    def __init__(self,
               annotations_file,
               audio_dir,
               transformation,
               target_sample_rate,
               time_length,
               rhythm_dict,
               max_length
               ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.time_length = time_length
        self.rhythm_dict = rhythm_dict
        self.max_length = max_length
        self.device = "cpu"

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
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
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
    
    def plot_spectrogram(self, specgram, title=None, ylabel="freq_bin", ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
        plt.show()

    def split_melspectrogram(self, signal):
        seq_len = math.ceil(signal.size(1) / self.time_length)
        total_size = signal.size(1)
        split_size = self.time_length
        remainder = total_size % split_size
        # print(total_size, seq_len, remainder)
        split_tensors = torch.split(signal, split_size, dim=1)

        # If there is a remainder, pad the last tensor
        if remainder > 0:
            last_tensor = split_tensors[-1]
            padding_size = split_size - last_tensor.size(1)
            last_tensor = torch.nn.functional.pad(last_tensor, (0, 0, 0, padding_size))
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
        signal_tensor = mel_signal.reshape(1, mel_signal.shape[2], mel_signal.shape[1])
        sequences = self.split_melspectrogram(signal_tensor)
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
        print(onehot_list[start:batch_size*batch])
        signal_batch = pad_sequence(signal_list[start:batch_size*batch], padding_value=-1,batch_first=True)
        print(signal_batch.shape)
        onehot_batch = pad_sequence(onehot_list[start:batch_size*batch], padding_value=0,batch_first=True)
        print(onehot_batch.shape)
        # for i in range(onehot_batch.size(0)):
        #     print(onehot_batch[i,:])
        label_batch = pad_sequence(label_list[start:batch_size*batch], padding_value=-1,batch_first=True)
        print(label_batch.shape)
        each_batch = (signal_batch, onehot_batch, label_batch)
        train_data_loader.append(each_batch)

        start = batch_size*batch
        


    # def my_collate(batch):
    #     # Sort the batch based on signal sequence length
    #     batch.sort(key=lambda x: x[0].size(0), reverse=True)

    #     # Separate signals, target_onehots, and target_labels
    #     signals, target_onehots, target_labels = zip(*batch)

    #     # Pad the signals dynamically
    #     padded_signals = pack_padded_sequence(torch.nn.utils.rnn.pad_sequence(signals, batch_first=True), 
    #                                         [signal.size(1) for signal in signals], 
    #                                         enforce_sorted=False)

    #     # Pad the target_onehots and target_labels using custom function
    #     padded_target_onehots, padded_target_labels = pad_targets(target_onehots, target_labels)
        

    #     return padded_signals, padded_target_onehots, padded_target_labels
    
    # def pad_targets(target_onehots, target_labels):
    #     max_length = max(target_onehot.size(0) for target_onehot in target_onehots)
    #     padded_target_onehots = torch.zeros(len(target_onehots), max_length, target_onehots[0].size(2))
    #     padded_target_labels = torch.zeros(len(target_labels), max_length)

    #     for i, (target_onehot, target_label) in enumerate(zip(target_onehots, target_labels)):
    #         length = target_onehot.size(0)
    #         padded_target_onehots[i, :length, :] = target_onehot.squeeze()
    #         padded_target_labels[i, :length] = target_label.squeeze()

    #     return padded_target_onehots, padded_target_labels
    

    # train_data_loader = DataLoader(data, batch_size, collate_fn = my_collate)

    return train_data_loader
    


if __name__ == "__main__":
    current_path = pathlib.Path(__file__).parent.resolve()
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    AUDIO_DIR = "dataset/01/09"
    ANNOTATIONS_FILE = AUDIO_DIR + "/metadata.csv"
    SAMPLE_RATE = 22050
    time_length = 100
    max_length = 60
    batch_size = 10

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 2048,
        # hop_length = 1025,
        n_mels = 64
        )


    sound = MusicDataset(ANNOTATIONS_FILE,
                    AUDIO_DIR,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    rhythm_dict,
                    max_length
                    )

    print(f"There are {len(sound)} samples in the dataset.")
    for i in range(len(sound)):
        signal,target_onehot, label = sound[i]
        print(signal.shape)

    train_data_loader = create_data_loader(sound, batch_size)

    print(len(train_data_loader))

    for batch in train_data_loader:
        signals, target_onehots, target_labels = batch
        print("Signals Shape:", signals.shape)
        print("Onehots Shape:", target_onehots.shape)
        print("Labels Shape:", target_labels.shape)



            
