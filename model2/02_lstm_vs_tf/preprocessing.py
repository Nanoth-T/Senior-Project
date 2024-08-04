### Import Library -----------------------------###
import torchaudio
import librosa

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import os
import pathlib
import ast

###---------------------------------------------###


class MusicDataset(Dataset):
    def __init__(self,
               annotations_file,
               audio_dir,
               transformation,
               target_sample_rate,
               time_length,
               overlap,
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
        self.overlap = overlap
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
        rt_indexes.append(self.rhythm_dict["<EOS>"])
        return torch.LongTensor(rt_indexes)
    
    def _targetInput(self, label):
        rt_indexes = [self.rhythm_dict[rt] for rt in label]
        rt_indexes.insert(0, self.rhythm_dict["<SOS>"])
        return torch.LongTensor(rt_indexes)
    
    def _normalize(self, signal):
        signal = F.normalize(signal)
        return signal
    
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
            print(last_tensor.shape)

        sequences = torch.stack(split_tensors)

        return sequences
    
    def split_overlap_melspectrogram(self, signal):
        step = self.time_length - self.overlap
        if signal.size(2) < self.time_length:
            signal = F.pad(signal, (0, self.time_length-signal.size(2)))
        else:
            reminder = signal.size(2) % self.overlap
            signal = F.pad(signal, (0, reminder))

        sequences = torch.unfold_copy(input=signal, dimension=2, size=self.time_length, step=step)
        sequences = sequences.permute(2, 0, 1, 3)
        return sequences

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label_tensor = self._targetTensor(label) 
        target_input = self._targetInput(label)

        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)

        mel_signal = self.transformation(signal)
        self.plot_spectrogram(mel_signal[0], f"{index}")
        # self.plot_spectrogram(signal_tensor[0])

        norm_signal = self._normalize(mel_signal)
        sequences_tensor = self.split_overlap_melspectrogram(norm_signal)

        # for s in range(len(sequences)):
        #     self.plot_spectrogram(sequences[s][0], f"{index}", f"{s+1}")

        sequences_tensor = torch.flatten(sequences_tensor, start_dim=1, end_dim=-1)


        return sequences_tensor, target_input, label_tensor


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

def collate_fn(batch):
    signal_list, targetinput_list, labeltensor_list = [], [], []

    # Extract data from each sample in the batch
    for signal, target_input, label_ten in batch:
        signal_list.append(signal)
        targetinput_list.append(target_input)
        labeltensor_list.append(label_ten)

    # Pad the sequences in each list
    signal_batch = pad_sequence(signal_list, padding_value=0, batch_first=True)
    targetinput_batch = pad_sequence(targetinput_list, padding_value=-1, batch_first=True)
    labeltensor_batch = pad_sequence(labeltensor_list, padding_value=-1, batch_first=True)

    return signal_batch, targetinput_batch, labeltensor_batch

if __name__ == "__main__":

    ### ------------ Variable ----------------- ###
    
    CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
    SAMPLE_RATE = 22050
    n_fft = 2048
    hop_length = 1025
    n_mels = 256

    rhythm_dict = {"<SOS>":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "rest_quarter": 6, "<EOS>":7}
    audio_dir = "dataset/prelim_data"
    ANNOTATIONS_FILE = audio_dir + "/metadata.csv"

    time_length = 200
    overlap = 50
    max_length = 20
    batch_size = 10
    

    ### -------------------------------------- ###



    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels = n_mels
        )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    sound = MusicDataset(ANNOTATIONS_FILE,
                    audio_dir,
                    mel_spectrogram,
                    SAMPLE_RATE,
                    time_length,
                    overlap,
                    rhythm_dict,
                    max_length,
                    device
                    )
    

    print(f"There are {len(sound)} samples in the dataset.")
    for i in range(len(sound)):
        signal, input, label = sound[i]
        # print(signal)

    train, test, idx_train, idx_test = train_test_split(sound, np.arange(len(sound)), test_size=0.3, random_state=0, shuffle=True)
    # print(train)
    print(idx_train)
    print("-----------------------")
    # print(test)
    print(idx_test)
    # train_data_loader = create_data_loader(sound, batch_size)

    train_data_loader = DataLoader(sound, batch_size=batch_size, collate_fn=collate_fn)
    print(len(train_data_loader))

    for batch in train_data_loader:
        signals, input, target_labels = batch
        # print(signals.max())
        # print("Signals Shape:", signals.shape)
        # print("input Shape:", input.shape, input)
        # print("Labels Shape:", target_labels.shape, target_labels)



            
