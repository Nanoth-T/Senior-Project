import streamlit as st
import pandas as pd
import numpy as np
import csv
import ast
import time
import plotly.express as px
# from streamlit_extras.stylable_container import stylable_container


rhyhtm_notation = {"whole": "webapp/pic/whole-note.png",
            "half": "webapp/pic/half-note.png",
            "quarter": "webapp/pic/quarter-note.png",
            "8th": "webapp/pic/eighth-note.png",
            "16th": "webapp/pic/sixteenth-note.png",
            "blank": "webapp/pic/blank.png"}

audio_path = {"Bella_Ciao": {"path" : "webapp/demo_realsomg_melody_fixed_velo/Bella_Ciao.wav", "caption":"tempo 120 BPM"},
              "Fur_Elise": {"path" : "webapp/demo_realsomg_melody_fixed_velo/Fur_Elise.wav", "caption":"tempo 120 BPM"},
              "Happy_Birthday": {"path" : "webapp/demo_realsomg_melody_fixed_velo/Happy_Birthday.wav", "caption":"tempo 120 BPM"},
              "Korobeiniki":{"path" : "webapp/demo_realsomg_melody_fixed_velo/Korobeiniki.wav", "caption":"tempo 125 BPM"},
              "London_Bridge_Is_Falling_Down":{"path" : "webapp/demo_realsomg_melody_fixed_velo/London_Bridge_Is_Falling_Down.wav", "caption":"tempo 160 BPM"},
              "Vande_Mataram_Traditional":{"path" : "webapp/demo_realsomg_melody_fixed_velo/Vande_Mataram_Traditional.wav", "caption":"tempo 100 BPM"}
              }

model_path = {
            # "LSTM" : None,
              "Unchunk-Transformer": "webapp/log/unchunk_demo_realsomg_melody_fixed_velo.csv",
              "Chunk-Transformer": "webapp/log/chunk1_demo_realsomg_melody_fixed_velo_FULL.csv"
}

def rhythm_accuracy(rhythm_predict, rhythm_target):

    if type(rhythm_predict) != list:
        rhythm_predict = ast.literal_eval(rhythm_predict)
    if type(rhythm_target) != list:
        rhythm_target = ast.literal_eval(rhythm_target)

    if len(rhythm_predict) == 0:
        return 0

    max_length = max(len(rhythm_predict), len(rhythm_target))
    if len(rhythm_predict) != len(rhythm_target):
        rhythm_predict.extend(["<PAD>"]*(max_length-len(rhythm_predict)))
        rhythm_target.extend(["<PAD>"]*(max_length-len(rhythm_target)))

    acc_rhythm = 0
    for i in range(max_length):
        if rhythm_predict[i] == rhythm_target[i]:
            acc_rhythm += 1
    
    acc_rhythm = acc_rhythm / max_length
    return acc_rhythm

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
                "audio_dir" : "webapp/demo_realsomg_melody_fixed_velo",
                'metadata_path' : "webapp/demo_realsomg_melody_fixed_velo/_metadata.csv",
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
            result = pd.read_csv(log_path)
            full_audio, target = result.loc[test_data.index, "decode_predict"].tolist()[0], result.loc[test_data.index, "decode_target"].tolist()[0]
            total_acc_rhythm = rhythm_accuracy(full_audio, target)
            full_audio, target = ast.literal_eval(full_audio), ast.literal_eval(target)

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
                                            st.image(rhyhtm_notation[target_actual_for_label[i + j]].replace(".png", "-blue.png"), caption=f"{target_actual[i + j]}", use_container_width=True,)
                                            # st.write(i + j, target[i + j])
                                        elif full_audio[i + j] != target_actual_for_label[i + j]:
                                            st.image(rhyhtm_notation[target_actual_for_label[i + j]].replace(".png", "-blue.png"), caption=f"{target_actual[i + j]}", use_container_width=True,)
                                            # st.write(i + j, target[i + j])
                                        else:
                                            st.image(rhyhtm_notation[target_actual_for_label[i + j]], caption=f"{target_actual[i + j]}", use_container_width=True)
                                            # st.write(i + j, target[i + j])
                            else:
                                with col:
                                    st.image(rhyhtm_notation["blank"], use_container_width=True)
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
                                    st.image(rhyhtm_notation["blank"], use_container_width=True)
                            else:
                                # i + j < len(full_audio):  # Check if there are remaining images
                                with col:
                                    if (i + j) > len(target)-1:
                                        st.image(rhyhtm_notation[full_audio[i + j]].replace(".png", "-red.png"), caption=f"{full_audio[i + j]}", use_container_width=True,)
                                            # st.write(i + j, target[i + j])
                                    elif full_audio[i + j] != target[i + j]:
                                        st.image(rhyhtm_notation[full_audio[i + j]].replace(".png", "-red.png"), caption=f"{full_audio[i + j]}", use_container_width=True,)
                                            # st.write(i + j, target[i + j])
                                    else:
                                        st.image(rhyhtm_notation[full_audio[i + j]], caption=f"{full_audio[i + j]}", use_container_width=True)
                                            # st.write(i + j, target[i + j])
                            # else:
                                
                        if i != range(0, len(full_audio), max_col)[-1]:
                            container.divider()
                    
                    fig = rhythm_pie_chart(full_audio, target_actual)
                    fig_list.append(fig)


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

            

