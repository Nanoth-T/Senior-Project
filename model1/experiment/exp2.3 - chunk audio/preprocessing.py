### Import Library -----------------------------###
import torchaudio
import librosa
import scipy.io.wavfile as wavfile

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import wandb
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import os
import pathlib
import ast
from deeprhythm import DeepRhythmPredictor


###---------------------------------------------###


class MusicDataset(Dataset):
    def __init__(self,
               annotations_file,
               audio_dir,
               transformation,
               target_sample_rate,
               time_length,
               overlap,
               output_dict,
               max_length,
               device,
               model_name = None,
               wandb=None,
               ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.time_length = time_length
        self.overlap = overlap
        self.output_dict = output_dict
        self.max_length = max_length
        self.model_name = model_name
        self.wandb = wandb

    def _get_audio_sample_path(self, index):
        # path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        path = self.annotations.iloc[index, 0]
        start_time = self.annotations.iloc[index, 1]
        end_time = self.annotations.iloc[index, 2]
        return path, start_time, end_time
    
    def getpath(self, index):
        return self.annotations.iloc[index, 0]

    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 3]
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
        all_note = self.output_dict["all_note"]
        rt_indexes = [int(rt) for rt in label]
        # rt_indexes.insert(0, int(all_note.index("<SOS>")))
        rt_indexes.append(int(all_note.index("<EOS>")))
        return torch.LongTensor(rt_indexes)
    
    def _targetInput(self, label):
        all_note = self.output_dict["all_note"]
        rt_indexes = [int(rt) for rt in label]
        rt_indexes.insert(0, int(all_note.index("<SOS>")))
        return torch.LongTensor(rt_indexes)
    
    def _normalize(self, signal):
        signal = F.normalize(signal)
        return signal
    
    def plot_spectrogram(self, specgram, title, subtitle=None, ylabel="freq_bin", ax=None):
        wandb = self.wandb
        # print(specgram.shape)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if subtitle is not None:
            title = f"{title}-{subtitle}"
        ax.set_title(f"Mel Spectrogram of sound {title}")
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram.cpu()), origin="lower", aspect="auto", interpolation="nearest")
        if wandb is not None:
            wandb.log({f'Mel Spectrogram/{title}': wandb.Image(fig)})
        
        plt.savefig(f"{title}.png")
        plt.close(fig)
        plt.show()
        return fig
   
    def split_overlap_melspectrogram(self, signal):
        step = self.time_length - self.overlap
        if signal.size(2) < self.time_length:
            signal = F.pad(signal, (0, self.time_length-signal.size(2)))
        else:
            reminder = signal.size(2) % step if self.overlap == 0 else signal.size(2) % self.overlap
            signal = F.pad(signal, (0, reminder))

        sequences = torch.unfold_copy(input=signal, dimension=2, size=self.time_length, step=step)
        sequences = sequences.permute(2, 0, 1, 3)
        return sequences

    def __getitem__(self, index):
        audio_sample_path, start_time, end_time = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label_tensor = self._targetTensor(label) 
        target_input = self._targetInput(label)

        signal, sr = torchaudio.load(audio_sample_path, normalize=True)
        # print(signal.shape)
        signal = signal[:, int(start_time*sr):int(end_time*sr)]
        # print(signal.shape)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)

        mel_signal = self.transformation(signal)
        self.plot_spectrogram(mel_signal[0], f"{index}")
        # self.plot_spectrogram(signal_tensor[0])
        norm_signal = self._normalize(mel_signal)
        sequences_tensor = self.split_overlap_melspectrogram(norm_signal)
        # for s in range(sequences_tensor.size(0)):
        #     self.plot_spectrogram(sequences_tensor[s][0], f"{index}", f"{s+1}")
        
        # sequences_tensor = sequences_tensor.permute(0, 1, 3, 2)

        # for s in range(sequences_tensor.size(0)):
        #     self.plot_spectrogram(sequences_tensor[s][0], f"permute{index}", f"{s+1}")

        if self.model_name in ['lstm', 'transformer_linear']:
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

def custom_collate_fn(output_dict):
    def collate_fn(batch):
        all_note = output_dict['all_note']
        signal_list, targetinput_list, labeltensor_list = [], [], []

        # Extract data from each sample in the batch
        for signal, target_input, label_ten in batch:
            signal_list.append(signal)
            targetinput_list.append(target_input)
            labeltensor_list.append(label_ten)

        # Pad the sequences in each list
        signal_batch = pad_sequence(signal_list, padding_value=0, batch_first=True)
        targetinput_batch = pad_sequence(targetinput_list, padding_value=all_note.index("<PAD>"), batch_first=True)
        labeltensor_batch = pad_sequence(labeltensor_list, padding_value=all_note.index("<PAD>"), batch_first=True)


        return signal_batch, targetinput_batch, labeltensor_batch
    return collate_fn

if __name__ == "__main__":

    ### ------------ Variable ----------------- ###
    
    CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
    SAMPLE_RATE = 22050
    n_fft = 2048
    hop_length = 1024
    n_mels = 256

    RANGE_NOTE_ON = 128
    RANGE_NOTE_OFF = 128
    RANGE_VEL = 32
    RANGE_TIME_SHIFT = 100
    midi_code = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL
    all_note = [i for i in range(midi_code)] + ["<SOS>", "<EOS>", "<PAD>"]
    n_note = len(all_note)
    output_dict = {"all_note":all_note, "n_note":n_note}
    audio_dir = "dataset/long_length_data"
    ANNOTATIONS_FILE = "dataset/long_length_data/train_metadata_chunks_fixed_time.csv"

    time_length = 10
    overlap = 0
    max_length = 20
    batch_size = 32
    

    ### -------------------------------------- ###



    MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels = n_mels
        )

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"Using device {DEVICE}")

    audio_dir = "dataset/sample_audio3_30note"
    train_metadata = "dataset/sample_audio3_30note/metadata_chunks.csv"
    test_metadata = "dataset/sample_audio3_30note/metadata_chunks.csv"
    validation_dir = None
    model_name = "transformer_linear"

    sound = MusicDataset(train_metadata,
                    audio_dir,
                    MEL_SPECTROGRAM,
                    SAMPLE_RATE,
                    time_length,
                    overlap,
                    output_dict,
                    max_length,
                    DEVICE,
                    model_name,
                    wandb=None
                    )
    

    print(f"There are {len(sound)} samples in the dataset.")
    # print(sound)
    # for i in range(len(sound)):
    #     signal, input, label = sound[i]
    #     # print(signal.shape,input, label)


    train_data_loader = DataLoader(sound, batch_size=batch_size, collate_fn=custom_collate_fn(output_dict))

    for batch in train_data_loader:
        signals, input, target_labels = batch
        # print(signals.max())
        print("Signals Shape:", signals.shape)
        print("input Shape:", input.shape)
        print("Labels Shape:", target_labels.shape)



            
