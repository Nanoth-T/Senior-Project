import streamlit as st
import torch, torchaudio
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import csv
import ast
import time
import matplotlib.pyplot as plt
import plotly.express as px
import pretty_midi
# from streamlit_extras.stylable_container import stylable_container

from preprocessing import MusicDataset_Chunk, MusicDataset_Unchunk, custom_collate_fn
from inference import Inference_Chunk, Inference_Unchunk, calculate_rhythm_note, rhythm_accuracy, sequence_accuracy, find_acc
from model import LSTMModel, TransformerConv2dModel, TransformerLinearModel

# import importlib.util
# # from model import TransformerLinearModel, PositionalEncoding
# unchunk_tf_pre = importlib.util.spec_from_file_location("preprocessing_unchunk_tf_pre", "08_midilike/preprocessing.py")
# preprocessing_unchunk_tf_pre = importlib.util.module_from_spec(unchunk_tf_pre)
# unchunk_tf_pre.loader.exec_module(preprocessing_unchunk_tf_pre)

# chunk_tf_pre = importlib.util.spec_from_file_location("preprocessing_chunk_tf_pre", "12_changerange/preprocessing.py")
# preprocessing_chunk_tf_pre = importlib.util.module_from_spec(chunk_tf_pre)
# chunk_tf_pre.loader.exec_module(preprocessing_chunk_tf_pre)

import sys, os
# sys.path.append(os.path.abspath('/home/bev/nanoth/0rhythm_tempo_pj2/12_changerange'))
# from inference import calculate_rhythm_note, sequence_accuracy, rhythm_accuracy, find_acc  # type: ignore
# # from preprocessing import MusicDataset_Chunk, custom_collate_fn as custom_collate_fn_12  # type: ignore

sys.path.append(os.path.abspath('/home/bev/nanoth/0rhythm_tempo_pj2/midi_processor_mod'))
from processor import encode_midi, decode_midi # type: ignore


rhyhtm_notation = {"whole": "@wabapp/pic/whole-note.png",
            "half": "@wabapp/pic/half-note.png",
            "quarter": "@wabapp/pic/quarter-note.png",
            "8th": "@wabapp/pic/eighth-note.png",
            "16th": "@wabapp/pic/sixteenth-note.png",
            "blank": "@wabapp/pic/blank.png"}

audio_path = {"Bella_Ciao": {"path" : "@wabapp/demo_realsomg_melody_fixed_velo/Bella_Ciao.wav", "caption":"tempo 120 BPM"},
              "Fur_Elise": {"path" : "@wabapp/demo_realsomg_melody_fixed_velo/Fur_Elise.wav", "caption":"tempo 120 BPM"},
              "Happy_Birthday": {"path" : "@wabapp/demo_realsomg_melody_fixed_velo/Happy_Birthday.wav", "caption":"tempo 120 BPM"},
              "Korobeiniki":{"path" : "@wabapp/demo_realsomg_melody_fixed_velo/Korobeiniki.wav", "caption":"tempo 125 BPM"},
              "London_Bridge_Is_Falling_Down":{"path" : "@wabapp/demo_realsomg_melody_fixed_velo/London_Bridge_Is_Falling_Down.wav", "caption":"tempo 160 BPM"},
              "Vande_Mataram_Traditional":{"path" : "@wabapp/demo_realsomg_melody_fixed_velo/Vande_Mataram_Traditional.wav", "caption":"tempo 100 BPM"}
              }

model_path = {
            # "LSTM" : None,
              "Unchunk-Transformer": "08_midilike/log/transformer_linear_unchunk_midilabel_0_20241103-0219/demo_realsomg_melody_fixed_velo.csv",
              "Chunk-Transformer": "@wabapp/log/transformer_linear_chunk_midilabel_0_20241104-1444/demo_realsomg_melody_fixed_velo_FULL.csv"
}


def predict(model, signal, max_length, all_note):

    # max_length = 10000
 
    model.eval()
    with torch.no_grad():
        decoder_outputs = model(signal, max_loop=max_length)
        decoder_output = decoder_outputs[0]

        predicted = []
        for elm in decoder_output:
            topv, topi = elm.topk(1)
            if topi.item() == all_note.index("<EOS>"):
                break
            elif topi.item() == all_note.index("<PAD>"):
                break
            predicted.append(all_note[topi.item()])


    return predicted


def Inference(model, config, m, log_path, addition):

    audio_dir = addition["audio_dir"]
    metadata_path = addition["metadata_path"]
    selected_metadata_path = pd.read_csv(metadata_path)
    selected_metadata_path = selected_metadata_path.loc[selected_metadata_path["name"] == addition["song_name"]]
    selected_metadata_path.to_csv(f"{audio_dir}/selected_song.csv", index=False)

    output_dict = config['output_dict']
    all_note = output_dict["all_note"]

    max_length = config['max_length']
    input_size = config["input_size"]

    MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
        sample_rate = config['sample_rate'],
        n_fft = config['n_fft'],
        hop_length = config['hop_length'],
        n_mels = config['n_mels']
        )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device {DEVICE}")
    
    model = config["model"]
    state_dict = torch.load(f"{log_path}/model.pth")   
    model.load_state_dict(state_dict['model_state_dict'])
    # model.eval()

    if m == "Chunk-Transformer":
        sound = MusicDataset_Chunk(annotations_file = f"{audio_dir}/selected_song.csv",
                            audio_dir = audio_dir,
                            transformation = MEL_SPECTROGRAM,
                            target_sample_rate = config['sample_rate'],
                            target_hop_length = config['hop_length'],
                            chunk_size = config['chunk_size'],
                            output_dict = config['output_dict'],
                            max_length = config['max_length'],
                            model_name = config['model_name'],
                            device = DEVICE)
        sound_data = DataLoader(sound, batch_size=1, collate_fn=custom_collate_fn(output_dict))
    elif m == "Unchunk-Transformer":
        sound = MusicDataset_Unchunk(annotations_file = f"{audio_dir}/selected_song.csv",
                         audio_dir = audio_dir,
                         transformation = MEL_SPECTROGRAM,
                         target_sample_rate = config['sample_rate'],
                         time_length = config['time_length'],
                         overlap = config['overlap'],
                         output_dict = config['output_dict'],
                         max_length = config['max_length'],
                         model_name = config['model_name'],
                         device = DEVICE)
    
        sound_data = DataLoader(sound, batch_size=1, collate_fn=custom_collate_fn(output_dict))
    test_full_data = pd.read_csv(f"{audio_dir}/selected_song.csv")

    data = []

    total_acc_all = []
    total_acc_time = []
    total_acc_rhythm = []

    row = 0
    for i, sound in enumerate(sound_data):
        # print(f"sound{i}------------------------------------------")
        
        if m == "Chunk-Transformer":
            signal = sound[0]
            metadata = sound[3]
            target = metadata[0]["midi_label"]
            tempo = metadata[0]["tempo"]
        elif m == "Unchunk-Transformer":
            signal = sound[0]
            target = sound[2].tolist()[0]
            tempo = sound[3][0]["tempo"]

        predicted  = predict(model, signal, max_length, all_note)
        # st.write(predicted)
        decode_predicted = decode_midi(predicted)

        for i in decode_predicted.instruments:
            decode_predicted_list = []
            for note in i.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end, tempo=tempo)
                decode_predicted_list.append(rhythm)

        expected = [all_note[i] for i in target]
        if expected[-1] == "<EOS>":
            expected.pop()
        decode_expected = decode_midi(expected)

        for j in decode_expected.instruments:
            decode_expected_list = []
            for note in j.notes:
                rhythm = calculate_rhythm_note(start_time=note.start, end_time=note.end, tempo=tempo)
                decode_expected_list.append(rhythm)

        # if decode_predicted_list != decode_expected_list:
        #     print(f"Predicted {len(decode_predicted_list)} items: {decode_predicted_list}")
        #     print(f"Expected {len(decode_expected_list)} items: {decode_expected_list}")
        if m == "Chunk-Transformer":
            data.append([expected, predicted, decode_expected_list, decode_predicted_list, str(metadata[0]["audio_idx"])])
        elif m == "Unchunk-Transformer":
            data.append([expected, predicted, decode_expected_list, decode_predicted_list, row])
        pred, target = predicted.copy(), expected.copy()
        acc_all, acc_time = sequence_accuracy(pred, target)
        acc_rhythm = rhythm_accuracy(rhythm_predict=decode_predicted_list.copy(), rhythm_target=decode_expected_list.copy())
        total_acc_all.append(acc_all)
        total_acc_time.append(acc_time)
        total_acc_rhythm.append(acc_rhythm)
        row += 1

    
    chunks = pd.DataFrame(data, columns=["encode_target", "encode_predict", "decode_target", "decode_predict", "audio_idx"])
    if m == "Chunk-Transformer":
        chunks = chunks.groupby(["audio_idx"], as_index = False, sort=False).agg('sum')
    full_audio = chunks["decode_predict"].to_list()[0]
    full_audio_list = full_audio.copy()
    target_audio = chunks["decode_target"].to_list()[0]
    target_audio_list = target_audio.copy()
    total_acc_all, total_acc_time, total_acc_rhythm = find_acc(chunks, test_full_data)
    return full_audio_list, target_audio_list, total_acc_all, total_acc_rhythm


### ---------------------- ###
# def predict(model, signal, max_length, all_note):

#     # max_length = 10000
 
#     model.eval()
#     with torch.no_grad():
#         st.write(signal)
#         decoder_outputs = model(signal, max_loop=max_length)
#         decoder_output = decoder_outputs[0]
#         predicted = []
#         for elm in decoder_output:
#             topv, topi = elm.topk(1)
#             if topi.item() == all_note.index("<EOS>"):
#                 break
#             elif topi.item() == all_note.index("<PAD>"):
#                 break
#             predicted.append(all_note[topi.item()])


#     return predicted
def rhythm_pie_chart(rhythm_predict, rhythm_target):

    if type(rhythm_predict) != list:
        rhythm_predict = ast.literal_eval(rhythm_predict)
    if type(rhythm_target) != list:
        rhythm_target = ast.literal_eval(rhythm_target)

    if len(rhythm_predict) == 0:
        acc_rhythm = 0

    max_length = max(len(rhythm_predict), len(rhythm_target))
    if len(rhythm_predict) != len(rhythm_target):
        rhythm_predict.extend(["<PAD>"]*(max_length-len(rhythm_predict)))
        rhythm_target.extend(["<PAD>"]*(max_length-len(rhythm_target)))

    acc_rhythm = 0
    for i in range(max_length):
        if rhythm_predict[i] == rhythm_target[i]:
            acc_rhythm += 1
    
    wrong = max_length - acc_rhythm
    temp = pd.DataFrame({
        "label": ["correct", "wrong"],
        "value": [acc_rhythm, wrong],
    })
    fig = px.pie(temp,
                 hole = 0.5,
                 values="value",
                 names="label",
                 color="label",
                 color_discrete_map={'wrong':'darkred',
                                 'correct':'darkblue',}
                )
                    
    return fig



## ------------------------- ##


st.title(":rainbow[Demo] $\\bigstar$")


# st.info('This is playground space for testing the models', icon="ℹ️")

st.subheader("Step 1 : Select models")

container = st.container(border=True)

models = container.multiselect(
    "**Select models for testing.**",
    list(model_path.keys()),
    help="All of these models were trained with labels that expand the range of time shifts. Additionally, the dataset contains 10-20 notes with tempos of 100, 120, and 125, totaling 5,000 samples overall",
    default=["Chunk-Transformer"]
)
container.caption(f"You selected: {', '.join(models)}")

st.subheader("Step 2 : Select song")
container = st.container(border=True)

col1, col2 = container.columns([2.5, 3])

with col1:
    song = st.radio(
        "**Select song for testing.**",
        tuple(audio_path.keys()),
        captions=[v["caption"] for v in audio_path.values()]
    )
        
with col2:
    st.write("")
    st.caption(f"You selected: {song}")
    for i in list(audio_path.keys()):
        if song == i:
            st.audio(audio_path[i]["path"])

col2_1, col2_2 = col2.columns([5, 1])

with col2_2:
    st.write("")
    st.write("")
    selected = st.button("Confirm")

if selected:
    # with st.spinner('Wait for it...'):
    #     time.sleep(2)
    st.text("")
    addition = {
                "audio_dir" : "@wabapp/demo_realsomg_melody_fixed_velo",
                'metadata_path' : "@wabapp/demo_realsomg_melody_fixed_velo/_metadata.csv",
                "song_name": song + ".wav",
                }
    
    max_col = 8

    metadata_path = addition["metadata_path"]
    test_data = pd.read_csv(metadata_path)
    test_data = test_data.loc[test_data["name"] == addition["song_name"]]
    target_actual = test_data["decode_rhythm"].apply(lambda x: ast.literal_eval(x)).to_list()[0]
    target_actual_for_label = target_actual.copy()

    main_container = st.container(border=False)
    # col1, col2 = main_container.columns(2)
    # col1.header(f"Target")
    # col2.header(f"Prediction")
    tabs = main_container.tabs(models + ["Compare"])
    

    acc_list = []
    fig_list = []
    
    with main_container:
        for m, tab in zip(models, tabs[:-1]):
            log_path = model_path[m]
            # state_dict = torch.load(f"{log_path}/model.pth")
            # config = state_dict["config_dict"]
            # model = None
            # full_audio, target, total_acc_all, total_acc_rhythm = Inference(model, config, m, log_path, addition)
            # i here target you edok thanks for remind me error. lol
            result = pd.read_csv(log_path)
            full_audio, target = result.loc[test_data.index, "decode_predict"].tolist()[0], result.loc[test_data.index, "decode_target"].tolist()[0]
            total_acc_rhythm = rhythm_accuracy(full_audio, target)
            full_audio, target = ast.literal_eval(full_audio), ast.literal_eval(target)
            # st.write(full_audio)
            # st.write(target)
            # st.write(total_acc_rhythm)
                
            # tab.subheader(f"{m}")
            # container = tab.container(border=True)
            with tab:
                target_col, pred_col = tab.columns(2)
                with target_col:
                    container = target_col.container(border=True)
                    container.header(f"Target", divider="gray")
                    container.write("")
                    for i in range(0, len(target_actual_for_label), max_col):
                        
                        cols = container.columns(max_col)
                        for j, col in enumerate(cols):
                            if i + j < len(target_actual_for_label):  # Check if there are remaining images
                                with col:
                                        if (i + j) > len(full_audio)-1:
                                            st.image(rhyhtm_notation[target_actual_for_label[i + j]].replace(".png", "-blue.png"), caption=f"{target_actual[i + j]}", use_column_width=True,)
                                            # st.write(i + j, target[i + j])
                                        elif full_audio[i + j] != target_actual_for_label[i + j]:
                                            st.image(rhyhtm_notation[target_actual_for_label[i + j]].replace(".png", "-blue.png"), caption=f"{target_actual[i + j]}", use_column_width=True,)
                                            # st.write(i + j, target[i + j])
                                        else:
                                            st.image(rhyhtm_notation[target_actual_for_label[i + j]], caption=f"{target_actual[i + j]}", use_column_width=True)
                                            # st.write(i + j, target[i + j])
                            else:
                                with col:
                                    st.image(rhyhtm_notation["blank"], use_column_width=True)
                        if i != range(0, len(target_actual_for_label), max_col)[-1]:
                            container.divider()

                        # Create columns for this row
                        # cols = st.columns(min(max_col, len(full_audio) - i))  # Adjust for remaining images
                with pred_col:
                    container = pred_col.container(border=True)
                    container.header(f"Prediction", divider="gray")
                    container.write("")
                    for i in range(0, len(full_audio), max_col):
                    
                        cols = container.columns(max_col)
                        for j, col in enumerate(cols):
                            if i + j >= len(full_audio) or full_audio[i+j] == "<PAD>":
                                with col:
                                    st.image(rhyhtm_notation["blank"], use_column_width=True)
                            else:
                                # i + j < len(full_audio):  # Check if there are remaining images
                                with col:
                                    if (i + j) > len(target)-1:
                                        st.image(rhyhtm_notation[full_audio[i + j]].replace(".png", "-red.png"), caption=f"{full_audio[i + j]}", use_column_width=True,)
                                            # st.write(i + j, target[i + j])
                                    elif full_audio[i + j] != target[i + j]:
                                        st.image(rhyhtm_notation[full_audio[i + j]].replace(".png", "-red.png"), caption=f"{full_audio[i + j]}", use_column_width=True,)
                                            # st.write(i + j, target[i + j])
                                    else:
                                        st.image(rhyhtm_notation[full_audio[i + j]], caption=f"{full_audio[i + j]}", use_column_width=True)
                                            # st.write(i + j, target[i + j])
                            # else:
                                
                        if i != range(0, len(full_audio), max_col)[-1]:
                            container.divider()
                    
                    fig = rhythm_pie_chart(full_audio, target_actual)
                    fig_list.append(fig)

                # container = st.container(border=True)
                # col1, col2= container.columns(2)
                # col1.metric(label=":green[Accuracy All]", value=total_acc_all,)
                acc_list.append({"label":f"{m}", "value":total_acc_rhythm})

        with tabs[-1]:
            cols = st.columns(len(fig_list))
            for a, f, c in zip(acc_list, fig_list, cols):
                # st.metric(label=a["label"], value=a['value'],)
                with c:
                    f.add_annotation(
                        text=a["label"],  
                        x=0.5, y=0.5,
                        font_size=16,
                        showarrow=False
                    )
                    f.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(f, use_container_width=True)

    st.toast('The result is ready!', icon='😍')

        # with target_col:
        #     container = st.container(border=True)
        #     tab = container.tabs([f"{song}"])
        #     with tab[0]:
        #         st.text("")
        #         for i in range(0, len(target_actual), max_col):
        #             # Create columns for this row
        #             # cols = st.columns(min(max_col, len(target) - i))  # Adjust for remaining images
        #             cols = st.columns(max_col)
        #             for j, col in enumerate(cols):
        #                 if i + j < len(target_actual):  # Check if there are remaining images
        #                     with col:
        #                         st.image(rhyhtm_notation[target_actual[i + j]], caption=f"{target_actual[i + j]}", use_column_width=True,)
        #                         # st.write(i + j)
        #                 else:
        #                     with col:
        #                         st.image(rhyhtm_notation["blank"], use_column_width=True)
        #             if i != range(0, len(target_actual), max_col)[-1]:
        #                 st.divider()

        # with pred_col:
            
        #     container = st.container(border=True)
        #     # tabs = container.tabs(models + ["Compare"])
            

