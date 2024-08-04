import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import ast
from torch.utils.tensorboard import SummaryWriter
import math
from sklearn.metrics import ConfusionMatrixDisplay

def InferencePlot(LOG_PATH, rhythm_to_id, WRITER):

    ID_TO_RHYTHM = {v: k for k, v in rhythm_to_id.items()}

    file = "all_data.csv"
    file_dir = f"{LOG_PATH}/{file}"

    data = pd.read_csv(file_dir)

    WRITER.add_figure(f'Excatly Accuracy Sequence', acc_sequence(data["predict"], data["target"]), 0)
    WRITER.add_figure(f'Correct Length Accuracy', correct_length(data["predict"], data["target"]), 0)

    wrong_index_ls, wrong_dict = wrong_index(data["predict"], data["target"])
    WRITER.add_figure(f'Incorrect idx', wrong_index_ls, 0)

    fig = plot_confusion_matrices(wrong_dict, rhythm_to_id)
    fig.savefig(f"{LOG_PATH}/cf_index.png")
    WRITER.add_figure(f'Incorrect Predict', fig , 0)
    WRITER.close()
    # writer.add_image(f'Incorrect Predict', f"{path}/cf_index.png" , 0)
    
    
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
            p.extend(["PAD"]*pad_num)

        for idx in range(len(t)):
            if idx not in wrong:
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
    plt.title("Incorrect Index")
    
    return fig, wrong

def plot_wrong(wrong, rhythm_to_id):
    fig = plt.figure(figsize=(10,5*math.ceil(len(wrong)/2)))
    for i in range(len(wrong)):
        plt.subplot(math.ceil(len(wrong)/2), 2, i+1)
        plt.title(f"Confusion Matrix at index {i}")
        ConfusionMatrixDisplay.from_predictions(wrong[i]["true"], wrong[i]["pred"], cmap=plt.cm.Blues, ax=plt.gca())
    return fig

def plot_confusion_matrices(wrong, rhythm_to_id):
    rows = math.ceil(len(wrong)/2)
    cols = 2
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 16))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3, hspace=0.5)

    for index, data in wrong.items():  # Iterate over dictionary entries
        true_labels, pred_labels = data["true"], data["pred"]  # Extract labels
        row, col = divmod(index, cols)  # Calculate row and col for subplot placement
        ConfusionMatrixDisplay.from_predictions(true_labels, pred_labels, cmap=plt.cm.Blues, ax=axes[row, col], labels=["whole", "half", "quarter", "8th", "<EOS>", "PAD"])
        axes[row, col].set_title(f"Confusion Matrix at index {index}")

    return fig




if __name__ == "__main__":

    rhythm_to_id = {"<SOS>":0, "whole": 1, "half": 2, "quarter": 3, "8th": 4, "16th": 5, "rest_quarter": 6, "<EOS>":7}
    ID_TO_RHYTHM = {v: k for k, v in rhythm_to_id.items()}

    path = "05_tfm_vs_lstm/rerun_exp_log/dataset/prelim_data_tf_1"
    file = "all_data.csv"
    file_dir = f"{path}/{file}"

    data = pd.read_csv(file_dir)

    Excatly_Accuracy_Sequence = acc_sequence(data["predict"], data["target"])
    Excatly_Accuracy_Sequence.savefig(f"{path}/Excatly_Accuracy_Sequence.png")

    Correct_Length_Accuracy = correct_length(data["predict"], data["target"])
    Correct_Length_Accuracy.savefig(f"{path}/Correct_Length_Accuracy.png")

    Incorrect_idx, wrong_dict = wrong_index(data["predict"], data["target"])
    Incorrect_idx.savefig(f"{path}/Incorrect_idx.png")

    cf_index = plot_confusion_matrices(wrong_dict, rhythm_to_id)
    cf_index.savefig(f"{path}/cf_index.png")

    
    print("completed")
