
import matplotlib.pyplot as plt
import os
from midiutil import MIDIFile
from midi2audio import FluidSynth
import random
import numpy as np
import pandas as pd
import subprocess



def fs(input, output):
  fluidsynth_executable = 'C:\\fluidsynth\\bin\\fluidsynth.exe'
  soundfont_path = 'C:\\Programdata\\soundfonts\\default.sf2'
  midi_file = f'C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name}\\{input}'
  output_file = f'C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name}\\{output}'

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
  os.remove(midi_file)


# Function to create a single track
def create_track(file_number):
    midi = MIDIFile(1)
    
    track = 0
    channel = 0
    start_time = 0.5
    volume = 100  # 0-127, as per the MIDI standard

    temp_tempo = [60, 70, 80, 90, 100, 110, 120]
    # Set tempo to 80 BPM
    tempo = random.choice(temp_tempo)
    midi.addTempo(track, start_time, tempo)

    # Define rhythm values
    rhythm_dict = {"whole": 4, "half": 2, "quarter": 1, "8th": 0.5, "16th": 0.25}

    # Create notes and durations for 4 bars (4 beats per bar in 4/4 time signature)
    beats_per_bar = 4
    # total_bars = random.randint(2, 4)
    total_bars = 4
    total_beats = beats_per_bar * total_bars
    rhythm_label = []
    # label = random.choice(list(rhythm_dict.keys()))
    while start_time < total_beats + 0.5:
        label = random.choice(list(rhythm_dict.keys()))
        duration = rhythm_dict[label]
            
        # Add the note (C4) at each duration
        pitch = 60  # MIDI note for C4
        midi.addNote(track, channel, pitch, start_time, duration, volume)
            
        start_time += duration  # Move to the next note/rest
        rhythm_label.append(label)

    with open(f"C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name}\\sound{file_number}.mid", "wb") as output_file:
        midi.writeFile(output_file)

    return rhythm_label

# Function to convert rhythm labels to integers
def rhythm_to_int(rhythm_label):
    rhythm_dict = {"whole": 0, "half": 1, "quarter": 2, "8th": 3, "16th": 4}
    return [rhythm_dict[label] for label in rhythm_label]


if __name__ == "__main__":
    
    folder_name = "01\\11"
    # Creating the DataFrame
    data = []
    number_sample = 100

    if not os.path.exists(f'C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name[:2]}'):
        os.makedirs(f'C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name[:2]}')
    if not os.path.exists(f'C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name}'):
        os.makedirs(f'C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name}')

    for i in range(number_sample):
        filename = f"sound{i}.wav"
        rhythm_label = create_track(i)

        # Convert midi file to audio wav file
        fs(f'sound{i}.mid', f'sound{i}.wav')

        # Convert rhythm labels to integers
        int_rhythms = rhythm_to_int(rhythm_label)
        
        data.append({"name": filename, "rhythm_label": rhythm_label, "target": int_rhythms})

    # Creating DataFrame
    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    df.to_csv(f"C:\\Users\\Administrator\\OneDrive - KMITL\\Desktop\\study\\!project\\dataset\\{folder_name}\\metadata.csv", index=False)
 
# "dataset\01\01"