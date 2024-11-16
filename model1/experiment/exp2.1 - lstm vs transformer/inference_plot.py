import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import ast
import wandb
import math
from sklearn.metrics import ConfusionMatrixDisplay
from training_loop import sequence_accuracy

def InferencePlot(LOG_PATH, output_dict, wandb):


    file = "test_data.csv"
    file_dir = f"{LOG_PATH}/{file}"

    data = pd.read_csv(file_dir)

    result_table = wandb.Table(dataframe=data)

    result_table_artifact = wandb.Artifact("result_table", type="result")
    result_table_artifact.add(result_table, "result_table")
    result_table_artifact.add_file(file_dir)

    wandb.log({"result_table": result_table})
    wandb.log_artifact(result_table_artifact)

    wandb.log({'Acc/Excatly Accuracy Sequence': wandb.Image(acc_sequence(data["decode_predict"], data["decode_target"]))})
    wandb.log({'Acc/Correct Length Accuracy': wandb.Image(correct_length(data["decode_predict"], data["decode_target"]))})

    wrong_index_ls, wrong_dict = wrong_index(data["decode_predict"], data["decode_target"])
    wandb.log({'Incorrect/Incorrect idx': wandb.Image(wrong_index_ls)})
    wandb.log({'Incorrect/Incorrect idx - norm': wandb.Image(plot_wrong_norm(wrong_dict))})

    cfm_idx, cfm_all = plot_confusion_matrices(wrong_dict)
    cfm_idx.savefig(f"{LOG_PATH}/cf_index.png")
    wandb.log({'Confusion Matrix/Index': wandb.Image(cfm_idx)})
    wandb.log({'Confusion Matrix/All': wandb.Image(cfm_all)})
    
    return "log completed"

def acc_sequence(pred, target):

    fig = plt.figure(figsize=(5,5))
    all_num = len(pred)
    number = 0
    for p, t in zip(pred, target):
        if ast.literal_eval(p) == ast.literal_eval(t):
            number += 1
    if all_num == number:
        plt.pie(x=[number], colors=["green"], labels=["correct"], autopct='%1.1f%%')
    else:
        plt.pie(x=[number, all_num-number], colors=["green", "red"], labels=["correct", "wrong"], autopct='%1.1f%%')
    return fig

def correct_length(pred, target):
    fig = plt.figure(figsize=(5,5))
    all_num = len(pred)
    number = 0
    for p, t in zip(pred, target):
        if len(ast.literal_eval(p)) == len(ast.literal_eval(t)):
            number += 1
    if all_num == number:
        plt.pie(x=[number], colors=["green"], labels=["correct"], autopct='%1.1f%%')
    else:
        plt.pie(x=[number, all_num-number], colors=["green", "red"], labels=["correct", "wrong"], autopct='%1.1f%%')
    return fig

def wrong_index(pred, target):
    fig, ax = plt.subplots(1, 1, figsize=(8,10))
    wrong = {}

    for p, t in zip(pred, target):
        p = ast.literal_eval(p)
        t = ast.literal_eval(t)

        if len(p) < len(t):
            pad_num = len(t) - len(p)
            p.extend(["<PAD>"]*pad_num)

        for idx in range(len(t)):
            if idx not in wrong.keys():
                wrong[idx] = {"true":[t[idx]],
                              "pred":[p[idx]],
                              "count":0}
            else:
                wrong[idx]["true"].append(t[idx])
                wrong[idx]["pred"].append(p[idx])

            if p[idx] != t[idx]:
                wrong[idx]["count"] += 1
        

    plt.bar_label(plt.bar(wrong.keys(), [wrong[i]["count"] for i in wrong.keys()]), fontsize=16)
    plt.xticks(list(wrong.keys()), fontsize=16)
    plt.xlabel("index", fontsize=16)
    plt.ylabel("amount", fontsize=16)
    plt.title("Incorrect Index - total")
    
    return fig, wrong

def plot_wrong_norm(wrong_dict):
    fig, ax = plt.subplots(1, 1, figsize=(8,10))
    plt.bar_label(plt.bar(wrong_dict.keys(), [wrong_dict[i]["count"] / len(wrong_dict[i]["true"]) for i in wrong_dict.keys()]), fontsize=16)
    plt.xticks(list(wrong_dict.keys()), fontsize=16)
    plt.xlabel("index", fontsize=16)
    plt.ylabel("amount", fontsize=16)
    plt.title("Incorrect Index - norm")

    return fig


def plot_confusion_matrices(wrong):
    rows = math.ceil(len(wrong)/2)
    cols = 2
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 24))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3, hspace=0.5)

    all_true = []; all_pred = []

    for index, data in wrong.items():  # Iterate over dictionary entries
        true_labels, pred_labels = data["true"], data["pred"]  # Extract labels
        all_true.extend(true_labels); all_pred.extend(pred_labels)
        row, col = divmod(index, cols)  # Calculate row and col for subplot placement
        ConfusionMatrixDisplay.from_predictions(true_labels, pred_labels, cmap=plt.cm.Blues, ax=axes[row, col], labels=["whole", "half", "quarter", "8th", "16th", "<EOS>", "<PAD>"])
        axes[row, col].set_title(f"Confusion Matrix at index {index}")
    plt.close()

    fig_all, ax = plt.subplots(1, 1, figsize=(8,10))
    ConfusionMatrixDisplay.from_predictions(all_true, all_pred, cmap=plt.cm.Blues, ax=ax, labels=["whole", "half", "quarter", "8th", "16th", "<EOS>", "<PAD>"])
    plt.title("Confusion Matrix")
    plt.close()

    return fig, fig_all




if __name__ == "__main__":


    rhythm_to_id = {"<SOS>":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "<PAD>": 6, "<EOS>":7}
    ID_TO_RHYTHM = {v: k for k, v in rhythm_to_id.items()}

    path = "07_tf_newlog/log/dataset/dif_tempo_same_note_data/dif_tempo_conv2d_emb20240813-1730"
    file = "all_data.csv"
    file_dir = f"{path}/{file}"

    data = pd.read_csv(file_dir)

    Excatly_Accuracy_Sequence = acc_sequence(data["predict"], data["target"])
    # Excatly_Accuracy_Sequence.savefig(f"{path}/Excatly_Accuracy_Sequence.png")

    Correct_Length_Accuracy = correct_length(data["predict"], data["target"])
    # Correct_Length_Accuracy.savefig(f"{path}/Correct_Length_Accuracy.png")

    Incorrect_idx, wrong_dict = wrong_index(data["predict"], data["target"])
    # Incorrect_idx.savefig(f"{path}/Incorrect_idx.png")

    cf_index = plot_confusion_matrices(wrong_dict, rhythm_to_id)
    cf_index.savefig(f"{path}/cf_index.png")
    # cf_index.savefig("07_tf_newlog/log/dataset/dif_length_data/dif_length_conv2d_emb20240813-1834/wandb/run-20240813_183413-x61y7uw1/files/media/images/Incorrect/Incorrect Predict_32007_060907f54fb2abdad4ac.png")

    
    print("completed")
