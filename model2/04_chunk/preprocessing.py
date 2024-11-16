### Import Library -----------------------------###
import torchaudio
import librosa
import scipy.io.wavfile as wavfile

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import Vol

import wandb
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import os
import pathlib
import ast

import sys; sys.path.append(os.path.abspath('/home/bev/nanoth/0rhythm_tempo_pj2/midi_processor_mod'))
from processor import encode_midi, decode_midi, Event # type: ignore
import pretty_midi

# from deeprhythm import DeepRhythmPredictor


###---------------------------------------------###


class MusicDataset(Dataset):
    def __init__(self,
               annotations_file,
               audio_dir,
               transformation,
               target_sample_rate,
               target_hop_length,
               chunk_size,
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
        self.target_hop_length = target_hop_length
        self.chunk_size = chunk_size
        self.output_dict = output_dict
        self.max_length = max_length
        self.model_name = model_name
        self.wandb = wandb

        self.chunk_metadata = self._prepare_chunks()

    def _adjust_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        if signal.shape[0] > 1: # (2, 1000) --> 2
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _prepare_chunks(self):
        """
        Prepare the chunk metadata for all audio files.
        This function generates the metadata (audio index, start time, etc.) for each chunk.
        """
        chunk_metadata = []
        
        # Loop over all audio files and calculate chunks
        for audio_idx, audio_file in enumerate(self.annotations["name"]):
            # tempo = self.annotations.loc[audio_idx, "tempo"]
            tempo = 120
            audio, sr = torchaudio.load(os.path.join(self.audio_dir, audio_file))
            audio = audio.to(self.device)
            audio = self._adjust_if_necessary(audio, sr)
            total_samples = audio.size(1)
            num_samples_per_chunk = int(self.chunk_size * self.target_sample_rate)
            num_chunks, reminder = divmod(total_samples, num_samples_per_chunk)
            # print(num_chunks, reminder)
            # num_chunks += 1 if reminder > 0 else num_chunks


            # Store metadata for each chunk
            for chunk_idx in range(num_chunks):
                # print(audio_idx, chunk_idx)
                start_sample = chunk_idx * num_samples_per_chunk
                end_sample = (chunk_idx + 1) * num_samples_per_chunk
                midi_label = self._prepare_midi_labels({'audio_idx': audio_idx, 'chunk_idx': chunk_idx, 
                                           'start_sample': start_sample, 'end_sample': end_sample})
                if len(midi_label) == 0:
                    continue
                chunk_metadata.append({
                        'audio_idx': audio_idx,
                        'chunk_idx': chunk_idx,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'midi_label': midi_label,
                        'tempo': tempo,
                        'is_last': False
                    })

            # Handle the last chunk (if it's shorter than the chunk size)
            if reminder > 0:
                start_sample = num_chunks * num_samples_per_chunk
                end_sample = total_samples
                midi_label = self._prepare_midi_labels({'audio_idx': audio_idx, 'chunk_idx': num_chunks, 
                                        'start_sample': start_sample, 'end_sample': end_sample})
                if len(midi_label) == 0:
                    continue
                else:
                    chunk_metadata.append({
                        'audio_idx': audio_idx,
                        'chunk_idx': num_chunks,
                        'start_sample': start_sample,
                        'end_sample': end_sample,  # This will be shorter than num_samples_per_chunk
                        'midi_label':midi_label,
                        'tempo':tempo,
                        'is_last': True  # Indicate it's the last chunk and might need padding
                    })

        return chunk_metadata
    
    def __len__(self):
        return len(self.chunk_metadata)
    
    def _prepare_midi_labels(self, chunk_metadata):
        """
        Prepare MIDI labels aligned with the chunk metadata.
        This function uses the chunk start/end times in seconds and extracts MIDI events that fit within each chunk.
        """      
        
        audio_idx = chunk_metadata['audio_idx']
        start_time = chunk_metadata['start_sample'] / self.target_sample_rate  # Convert start sample to time (seconds)
        end_time = chunk_metadata['end_sample'] / self.target_sample_rate      # Convert end sample to time (seconds)

            
        # Get the MIDI file corresponding to the audio file
        midi_file_path = os.path.join(self.audio_dir, self.annotations.iloc[audio_idx]['name'].replace(".wav", ".mid"))
        midi_data = encode_midi(midi_file_path)  # Call your MIDI encoding method
            
        # Filter MIDI events based on the chunk's start and end times
        filtered_midi_events = []
        current_time = 0
        for event in range(len(midi_data)):
            event_obj = Event.from_int(midi_data[event])
                
        # If the event is a time shift, update the current time
            if event_obj.type == 'time_shift':
                current_time += (event_obj.value + 1) / 100  # Assuming time shift is in 10ms units

                # Include events that occur within the chunk's time range
            if chunk_metadata["chunk_idx"] != 0 and start_time < current_time:
                filtered_midi_events.append(midi_data[event])
            
            if chunk_metadata["chunk_idx"] == 0 and start_time <= current_time:
                filtered_midi_events.append(midi_data[event])
                
            # Stop processing once past the chunk's end time
            if current_time > end_time:
                filtered_midi_events.append(midi_data[event+1])
                break
        if chunk_metadata["chunk_idx"] != 0:
            filtered_midi_events = filtered_midi_events[2:]
        return filtered_midi_events

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
   

    def __getitem__(self, index):

        

        # Get the chunk metadata
        chunk_info = self.chunk_metadata[index]
        audio_idx = chunk_info['audio_idx']
        start_sample = chunk_info['start_sample']
        end_sample = chunk_info['end_sample']
        label = chunk_info['midi_label']

         # Load the corresponding audio
        audio, sr = torchaudio.load(os.path.join(self.audio_dir, self.annotations.loc[audio_idx, "name"]))
        audio = audio.to(self.device)
        audio = self._adjust_if_necessary(audio, sr)

        chunk_audio = audio[:, start_sample:end_sample]

        # Handle padding for the last chunk if necessary
        if chunk_info['is_last']:
            num_samples_per_chunk = int(self.chunk_size * self.target_sample_rate)
            if chunk_audio.size(1) < num_samples_per_chunk:
                # Pad the audio with zeros to match the chunk size
                padding = num_samples_per_chunk - chunk_audio.size(1)
                chunk_audio = F.pad(chunk_audio, (0, padding))

        chunk_audio = chunk_audio.to(self.device)
        # Generate mel-spectrogram for the chunk
        mel_spectrogram = self.transformation(chunk_audio)
        mel_spectrogram = self._normalize(mel_spectrogram)
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1)
        target_input = self._targetInput(label)
        label_tensor =self._targetTensor(label)
        

        return mel_spectrogram, target_input, label_tensor, chunk_info

def custom_collate_fn(output_dict):
    def collate_fn(batch):
        all_note = output_dict['all_note']
        signal_list, targetinput_list, labeltensor_list, chunk_info_list = [], [], [], []

        # Extract data from each sample in the batch
        for signal, target_input, label_ten, chunk_info in batch:
            signal_list.append(signal)
            targetinput_list.append(target_input)
            labeltensor_list.append(label_ten)
            chunk_info_list.append(chunk_info)

        # Pad the sequences in each list
        signal_batch = pad_sequence(signal_list, padding_value=0, batch_first=True)
        targetinput_batch = pad_sequence(targetinput_list, padding_value=all_note.index("<PAD>"), batch_first=True)
        labeltensor_batch = pad_sequence(labeltensor_list, padding_value=all_note.index("<PAD>"), batch_first=True)

        

        return signal_batch, targetinput_batch, labeltensor_batch, chunk_info_list
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
    RANGE_VEL = 128
    RANGE_TIME_SHIFT = 500
    midi_code = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL
    all_note = [i for i in range(midi_code)] + ["<SOS>", "<EOS>", "<PAD>"]
    n_note = len(all_note)
    output_dict = {"all_note":all_note, "n_note":n_note}
    audio_dir = "dataset/long_length_data"
    ANNOTATIONS_FILE = "dataset/long_length_data/train_metadata2.csv"

    chunk_size = 5
    max_length = 200
    batch_size = 2
    

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

    audio_dir = "@wabapp/demo_realsomg_melody_fixed_velo2/"
    train_metadata = "@wabapp/demo_realsomg_melody_fixed_velo2/_metadata.csv"
    test_metadata = "@wabapp/demo_realsomg_melody_fixed_velo2/_metadata.csv"
    validation_dir = None
    model_name = "transformer_linear"

    sound = MusicDataset(train_metadata,
                    audio_dir,
                    MEL_SPECTROGRAM,
                    SAMPLE_RATE,
                    hop_length,
                    chunk_size,
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


    train_data_loader = DataLoader(sound, batch_size=10, collate_fn=custom_collate_fn(output_dict))


    for batch in train_data_loader:
        signals, input, target_labels, chunk_info = batch
        # print(signals.max())
        # print("Signals Shape:", signals.shape)
        # print("input Shape:", input.shape)
        for i in chunk_info:
            print(i)
            # print(i["midi_label"])
        # print("Labels Shape:", target_labels.shape)
    




            
