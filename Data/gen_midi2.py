import matplotlib.pyplot as plt
import os
import pretty_midi
from midi2audio import FluidSynth
import random
import numpy as np
import pandas as pd
import subprocess

from midi_processor.processor import encode_midi, decode_midi



def fs(folder_name, input, output, remove_midi=False):
  fluidsynth_executable = 'C:\\FluidSynth\\bin\\fluidsynth.exe'
  soundfont_path = 'C:\\ProgramData\\soundfonts\\default.sf2'
  midi_file = f'C:\\Users\\Lenovo\\OneDrive\\Desktop\\study\\!project\\{folder_name}\\{input}'
  output_file = f'C:\\Users\\Lenovo\\OneDrive\\Desktop\\study\\!project\\{folder_name}\\{output}'

  # Construct the command
  command = [
      fluidsynth_executable,
      '-i',
      soundfont_path,
      '-F', output_file,
      midi_file
  ]

  try:
      subprocess.run(command, check=True)
  except subprocess.CalledProcessError as e:
      print(f"Error: {e}")

  # delete midi file : keep only audio file
  if remove_midi == True:
    os.remove(midi_file)


def create_track_rhythm(file_name, rhythm_label, velo, tempo):
    midi_data = pretty_midi.PrettyMIDI(initial_tempo = tempo)
    piano = pretty_midi.Instrument(program=0)
    midi_data.instruments.append(piano)

    start_time = 0
    
    pitch_list = range(30, 100)
    # vel_list = range(1, 128)
    # vel_list = [127]
    # pitch_list = [47, 48, 50, 52, 53, 55, 57, 58, 60, 67, 69, 71, 72, 74, 76, 77, 79]

    for note in rhythm_label:
        # pitch = random.randint(50, 80)
        pitch = random.choice(pitch_list)
        velocity = velo
        duration = (rhythm_durations[note] * 60) / tempo
        end_time = start_time + duration
        piano.notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=end_time
        ))
        start_time = end_time


    midi_data.write(f"C:\\Users\\Lenovo\\OneDrive\\Desktop\\study\\!project\\{folder_name}\\{file_name}.mid")

    return rhythm_label

def create_track_time(file_name, rhythm_durations, tempo, velocity):
    midi_data = pretty_midi.PrettyMIDI(initial_tempo = tempo)
    piano = pretty_midi.Instrument(program=0)
    midi_data.instruments.append(piano)

    start_time = 0
    
    # pitch_list = range(21, 109)
    pitch_list = [47, 48, 50, 52, 53, 55, 57, 58, 60, 67, 69, 71, 72, 74, 76, 77, 79]
    rhythm_label = []

    while start_time < 60:
        # pitch = random.randint(50, 80)
        pitch = random.choice(pitch_list)
        rhythm = random.choice(list(rhythm_durations.keys()))
        rhythm_label.append(rhythm)
        duration = (rhythm_durations[rhythm] * 60) / tempo
        end_time = start_time + duration
        piano.notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=end_time
        ))
        start_time = end_time


    midi_data.write(f"C:\\Users\\Lenovo\\OneDrive\\Desktop\\study\\!project\\{folder_name}\\{file_name}.mid")

    return rhythm_label

def calculate_rhythm_note(start_time, end_time, tempo=120):
    # print(start_time, end_time)
    duration_seconds = round(end_time - start_time, 2)

    beats_per_second = round(tempo / 60, 2)
    beats = round(duration_seconds * beats_per_second, 1)

    # print(duration_seconds, beats)
    
    # Determine the rhythm note based on the number of beats
    if beats > 3: # 4
        return "whole"
    elif 1.5 < beats <= 3: # 2
        return "half"
    elif 0.75 < beats <= 1.5: # 1
        return "quarter"
    elif 0.375 < beats <= 0.75: # 0.5
        return "8th"
    elif 0.25 < beats <= 0.375: # 0.25
        return "16th"
    else:
        return "16th"

import itertools
def all_possible_list(rhythm_durations):
    all_list = list(itertools.product(rhythm_durations.keys(), repeat=len(rhythm_durations)))
    return all_list

def fixed_length(rhythm_durations, length, number_sample=1):
    all_list = []
    for _ in range(number_sample):
        random_list = random.choices(list(rhythm_durations.keys()), k=length)
        all_list.append(random_list)
    return all_list

def random_length(rhythm_durations, length_list, number_sample=1):
    all_list = []
    for _ in range(number_sample):
        random_list = random.choices(list(rhythm_durations.keys()), k=random.choice(length_list))
        all_list.append(random_list)
    return all_list



if __name__ == "__main__":
    
    folder_name = "train_demo_fixed_velo80"
    # Creating the DataFrame
    data = []
    # if not os.path.exists(f'C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name[:2]}'):
    #     os.makedirs(f'C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name[:2]}')
    if not os.path.exists(f'C:\\Users\\Lenovo\\OneDrive\\Desktop\\study\\!project\\{folder_name}'):
        os.makedirs(f'C:\\Users\\Lenovo\\OneDrive\\Desktop\\study\\!project\\{folder_name}')

    rhythm_durations = {"whole": 4.0, "half": 2.0, "quarter": 1.0, "8th": 0.5, "16th": 0.25}

    number_sample = 5000

    # label_list = all_possible_list(rhythm_durations)
    # label_list = fixed_length(rhythm_durations, length=10, number_sample=1)
    label_list = random_length(rhythm_durations, length_list=range(10, 21), number_sample=number_sample)

    tempo_list = [100, 120, 125]
    # tempo_list = [120, 110, 100]
    # vel_list = [49, 64, 70, 75, 80, 100]
    vel_list = [80]
    

    # for s in range(len(label_list)):
    for s in range(number_sample):
        rhythm_label = list(label_list[s])
        tempo = random.choice(tempo_list)
        velo = random.choice(vel_list)
        # tempo = 120
        filename = f"sound{s}"
        gen_rhythm = create_track_rhythm(f"{filename}", rhythm_label, velo, tempo)
        encode_label = encode_midi(os.path.join(f"{folder_name}/", f"{filename}.mid"))
        decoded = decode_midi(encode_label)
        for i in decoded.instruments:
            decode_rhythm = []
            for note in i.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end, tempo=tempo)
                decode_rhythm.append(rhythm)
                        
        # Convert midi file to audio wav file
        fs(folder_name ,f'{filename}.mid', f'{filename}.wav', remove_midi=False)

        split = "test" if random.random() >= 0.7 else "train"

        if decode_rhythm != gen_rhythm:
            print(decode_rhythm, gen_rhythm)
            break
        else:
            data.append({"name": f"{filename}.wav", "encode_label": encode_label, "decode_rhythm":decode_rhythm, "gen_rhythm":gen_rhythm, "tempo":tempo, "split": split})

        


    # Creating DataFrame
    metadata = pd.DataFrame(data)
    # metadata.to_csv(f"{folder_name}/metadata.csv", index=False)

    train_metadata = metadata.loc[metadata["split"] == "train"]
    train_metadata.to_csv(f"{folder_name}/train_metadata.csv", index=False)

    # validation_metadata = metadata.loc[metadata["split"] == "validation"]
    # validation_metadata.to_csv(f"{folder_name}/validation_metadata.csv", index=False)

    test_metadata = metadata.loc[metadata["split"] == "test"]
    test_metadata.to_csv(f"{folder_name}/test_metadata.csv", index=False)