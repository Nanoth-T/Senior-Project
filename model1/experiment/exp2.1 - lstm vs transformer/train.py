import pathlib, shutil, time
import numpy as np
import pandas as pd
import wandb
import tempfile
import os
import torch, torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
import ast
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import Counter

from preprocessing import MusicDataset, custom_collate_fn
from training_loop import run_training
from inference import Inference
from inference_plot import InferencePlot
from model import LSTMModel, TransformerConv2dModel, TransformerLinearModel

def main():
    ### ------------ Variable / Parameter ----------------- ###
    force_start = "YES"

    CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
    SAMPLE_RATE = 22050
    n_fft = 2048
    hop_length = 1025
    n_mels = 256

    time_length = 10
    overlap = 5
    max_length = 60
    batch_size = 32
    input_size = n_mels*time_length

    lr = 0.0001
    num_epochs = 500

    RANGE_NOTE_ON = 128
    RANGE_NOTE_OFF = 128
    RANGE_VEL = 32
    RANGE_TIME_SHIFT = 100

    # note_on_token = [f'<Event type: note_on, value:{j}>' for j in range(0, RANGE_NOTE_ON)]
    # note_off_token = [f'<Event type: note_off, value:{j}>' for j in range(0, RANGE_NOTE_OFF)]
    # time_token = [f'<Event type: time_shift, value: {i}>' for i in range(RANGE_TIME_SHIFT)]
    # velocity = [f'<Event type: velocity, value: {i}>' for i in range(RANGE_VEL)]
    rhythm_note = ["whole", "half", "quarter", "8th", "16th"]
    all_note = [i for i in rhythm_note] + ["<SOS>", "<EOS>", "<PAD>"]
    n_note = len(all_note)
    output_dict = {"all_note":all_note, "n_note":n_note}

    audio_dir = "dataset/diff_5to7note_data"
    train_metadata = "dataset/diff_5to7note_data/train_metadata.csv"
    test_metadata = "dataset/diff_5to7note_data/test_metadata.csv"
    validation_dir = None
    model_name = "transformer_linear" # ['lstm', 'transformer_linear', 'transformer_conv2d']
    log_tags = [model_name, audio_dir]
    log_notes = "edit some config"
    random_state = 1
    experiment_name = model_name + '_' + audio_dir + '_' + str(random_state)

    #------------------------------------------------------#

    # Default variable with condition #
    if validation_dir is None: 
        validation_dir = audio_dir 
        validate_size = 0.3
    if experiment_name is None: 
        experiment_name = "Temp"
    

    WANDB_PATH = 'tempfile'
    LOG_PATH = f'{str(CURRENT_PATH)}/log/{audio_dir}/{str(experiment_name)}{time.strftime("%Y%m%d-%H%M", time.localtime())}'
    directory_path = pathlib.Path(LOG_PATH)
    directory_path.mkdir(parents=True, exist_ok=True)

    MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = n_fft,
        hop_length = hop_length,
        n_mels = n_mels
        )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device {DEVICE}")

    train_set = MusicDataset(train_metadata,
                    audio_dir,
                    MEL_SPECTROGRAM,
                    SAMPLE_RATE,
                    time_length,
                    overlap,
                    output_dict,
                    max_length,
                    DEVICE,
                    model_name,
                    wandb
                    )
    train_set, validation_set, idx_train, idx_validation = train_test_split(train_set, np.arange(len(train_set)), test_size=validate_size, random_state=random_state, shuffle=True)

    print(f"There are {len(train_set)} samples in the dataset.")

    global model_config
    model_config = {
    'lstm': (LSTMModel, {'input_size':input_size, 'hidden_size':128, 'output_dict':output_dict, 'max_length':max_length}),
    'transformer_linear': (TransformerLinearModel, {'input_size':input_size, 'hidden_size' : 128, 'output_dict':output_dict, 'max_length':max_length,
                                                     'encoder_heads' : 1, 'encoder_layers' : 1,
                                                     'decoder_heads' : 1, 'decoder_layers' : 1}),
    'transformer_conv2d': (TransformerConv2dModel, {'input_size':input_size, 'hidden_size' : 128, 'output_dict':output_dict, 'max_length':max_length,
                                                     'encoder_heads' : 1, 'encoder_layers' : 1,
                                                     'decoder_heads' : 1, 'decoder_layers' : 1})
    }

    config={
        "model_name": model_name,
        "dataset" : audio_dir,
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        'sample_rate': SAMPLE_RATE, 'n_fft':n_fft, 'hop_length': hop_length, 'n_mels':n_mels,
        'time_length':time_length, 'overlap':overlap, 'max_length': max_length, 
        'input_size':input_size, 'output_dict': output_dict, 'random_state':random_state}
    
    # Initialize WandB
    wandb.init(project="exp1-lstm-vs-transformer-repeat", config=config,
               dir=WANDB_PATH, name=experiment_name, tags=log_tags, notes=log_notes)
    
    wandb.config.update(model_config[model_name][1])

    visualize_data(train_metadata, idx_train, idx_validation, test_metadata, wandb)


    model = get_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    other_config = {'experiment_name' : experiment_name,
                    'model' : model,
                    'criterion':criterion,
                    'model_optimizer':model_optimizer,
                    'device':DEVICE,
                    'log_path':LOG_PATH,
                    'audio_dir':audio_dir,
                    'train_metadata':train_metadata,
                    'test_metadata':test_metadata}
    
    config.update(other_config)

    print("Recheck Your Parameter")
    print(config)
    print("You want to continue?")
    force_start = str(input("YES|NO : ")) if force_start is None else "YES"
    if (force_start == "YES" ):
        print("OKiee, about to start...")

        
        run_training(model, train_set, validation_set, config, wandb)
        print(Inference(model, config, LOG_PATH,addition = None, wandb = wandb))
        print(InferencePlot(LOG_PATH, output_dict, wandb))
    
        wandb.finish()
    else:
        shutil.rmtree(LOG_PATH)


def get_model(model_name):
    model_class, params = model_config.get(model_name, (None, None))
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found")
    return model_class(**params)

### Use for Logging ----------------------------------------------------###

def plot_dist(name, label, rhythm_note):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    all_label = [j for i in label for j in ast.literal_eval(i)]
    counts = Counter(all_label)
    sorted_labels = sorted(counts.items(), key=lambda x: rhythm_note.index(x[0]))
    labels_sorted, values_sorted = zip(*sorted_labels)
    plt.bar(labels_sorted, values_sorted)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(f'Count of Each Label in {name}')
    return fig

def visualize_data(train_metadata, idx_train, idx_val, test_metadata, wandb):
    rhythm_note = ["whole", "half", "quarter", "8th", "16th"]
    # train set
    train_label = pd.read_csv(train_metadata)
    train_label = train_label.iloc[idx_train.tolist()]["decode_rhythm"].to_list()

    # validation set
    validation_label = pd.read_csv(train_metadata)
    validation_label = validation_label.iloc[idx_val.tolist()]["decode_rhythm"].to_list()

    # test set
    test_label = pd.read_csv(test_metadata)
    test_label = test_label["decode_rhythm"].to_list()

    wandb.log({f'Frequency Distribution/Train': wandb.Image(plot_dist(f"Train Set", train_label, rhythm_note)),
               f'Frequency Distribution/Validation': wandb.Image(plot_dist(f"Validation Set", validation_label, rhythm_note)),
               f'Frequency Distribution/Test': wandb.Image(plot_dist(f"Test Set", test_label, rhythm_note))})


if __name__ == "__main__":
    model_config = {}
    main()
