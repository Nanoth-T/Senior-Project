import torchaudio
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence


class MusicDataset(Dataset):
    def __init__(self,
               annotations_file,
               audio_dir,
               transformation,
               target_sample_rate,
               rhythm_dict,
               seq_len, 
               overlap
               ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = self._calculate_max_num_sample()
        self.rhythm_dict = rhythm_dict
        self.device = "cpu"
        self.seq_len = seq_len
        self.overlap = overlap

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 1]
        label = ast.literal_eval(label)
        return label
    
    def __len__(self):
        return len(self.annotations)
    
    def _calculate_max_num_sample(self):
        num_samples = 0
        for file_path in range(len(self.annotations)):
            audio_tensor, _ = torchaudio.load(os.path.join(self.audio_dir, self.annotations.iloc[file_path, 0]))
            num_samples = max(num_samples, audio_tensor.shape[1])
        return num_samples
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # (2, 1000) --> 2
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
        # [1, 1, 1] -> [1, 1, 1, 0, 0] zero right pading
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) # (left pad, right pad)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _onset_detection(self, signal):
        audio_np = signal.numpy()[0]  # Convert to NumPy array
        onset_frames = librosa.onset.onset_detect(y=audio_np, sr=self.target_sample_rate)
        # Create feature representation combining spectrogram and onset information
        # onset_tensor = torch.zeros(mel_signal.shape[2])  # Initialize onset tensor
        # onset_tensor[onset_frames] = 1  # Set values corresponding to onsets to 1
        return onset_frames
    
    def _targetTensor(self, label):
        rt_indexes = [self.rhythm_dict[rt] for rt in label]
        rt_indexes.append(self.rhythm_dict["EOS"]) # EOS
        # rt_indexes.insert(0, self.rhythm_dict["SOS"])
        while len(rt_indexes) < 60:
            rt_indexes.append(-1)
        return torch.LongTensor(rt_indexes)
    
    def _target_onehot(self, label):
        target_onehot_tensor = torch.zeros(60, 1, len(self.rhythm_dict))
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
        total_size = signal.size(1)
        split_size = total_size // self.seq_len
        remainder = total_size % self.seq_len

        split_tensors = torch.split(signal, split_size, dim=1)

        # If there is a remainder, pad the last tensor
        if remainder > 0:
            padding_size = split_size - (total_size - (self.seq_len - 1) * split_size)
            last_tensor = torch.nn.functional.pad(split_tensors[-1], (0, padding_size))
            split_tensors = split_tensors[:-1] + (last_tensor,)

        return split_tensors

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label_tensor = self._targetTensor(label) 
        target_onehot = self._target_onehot(label) 
        signal, sr = torchaudio.load(audio_sample_path) # ทุก signal ไม่ได้มี sr เท่ากัน เพราะงั้นต้อง resample ให้ sr มันเท่ากันด้วย
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        # print(signal.shape)
        #ต้องทำให้มี period sample เท่ากัน
        signal = self._cut_if_necessary(signal) # ถ้ามากกว่าต้องตัด
        signal = self._right_pad_if_necessary(signal) # ถ้ามันน้อยกว่าต้องเพิ่ม
        # print(signal.shape)

        mel_signal = self.transformation(signal) #**** take waveform to transformation
        # self.plot_spectrogram(mel_signal[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
        # onset_frames = self._onset_detection(mel_signal)
        # print(torch.flatten(mel_signal).shape)
        # mel_signal = mel_signal.unsqueeze(0)
        # signal_tensor = torch.cat((mel_signal, onset_tensor.unsqueeze(0).unsqueeze(0).expand_as(mel_signal)), dim=1)
        signal_tensor = mel_signal.reshape(1, mel_signal.shape[2], mel_signal.shape[1])
        sequences = self.split_melspectrogram(signal_tensor)
        sequences_tensor = torch.stack(sequences)


        return sequences_tensor, target_onehot, label_tensor



if __name__ == "__main__":
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4, "EOS":5, "SOS":6,}
    ANNOTATIONS_FILE = "dataset/02-rhythm/metadata.csv"
    AUDIO_DIR = "dataset/02-rhythm"
    SAMPLE_RATE = 22050
    seq_len = 5
    overlap = 0
    # NUM_SAMPLES = 607744

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 2048,
        hop_length = 1025,
        n_mels = 256
        )
    
    spectrogram = torchaudio.transforms.Spectrogram(n_fft = 1024)

    sound = MusicDataset(ANNOTATIONS_FILE,
                    AUDIO_DIR,
                    mel_spectrogram,
                    # spectrogram,
                    SAMPLE_RATE,
                    # NUM_SAMPLES,
                    rhythm_dict,
                    seq_len, 
                    overlap
                    )

    print(f"There are {len(sound)} samples in the dataset.")
    temp_list = []
    tg_list = []
    lb_list = []
    for i in range(len(sound)):
        signal,target_onehot, label = sound[i]
        temp_list.append(signal)
        tg_list.append(target_onehot)
        lb_list.append(label)
        print(signal.shape)
        print(target_onehot.size())
        print(label.size())

    train_batch10 = pad_sequence(temp_list, padding_value=-1, batch_first=True)
    tg_batch10 = pad_sequence(tg_list, padding_value=0, batch_first=True)
    lb_batch10 = pad_sequence(lb_list, padding_value=-1, batch_first=True)
    print(train_batch10.size())
    print(tg_batch10.size())
    # print(tg_batch10)
    print(lb_batch10.size())
    # print(lb_batch10)

            
