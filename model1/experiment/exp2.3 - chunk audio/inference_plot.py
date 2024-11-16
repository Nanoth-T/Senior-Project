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

    wandb.log({'Acc/Excatly Accuracy Sequence': wandb.Image(acc_sequence(data["decode_predict"], data["decode_target"]))})
    wandb.log({'Acc/Correct Length Accuracy': wandb.Image(correct_length(data["decode_predict"], data["decode_target"]))})

    wrong_index_ls, wrong_dict = wrong_index(data["decode_predict"], data["decode_target"])
    wandb.log({'Incorrect/Incorrect idx': wandb.Image(wrong_index_ls)})
    wandb.log({'Incorrect/Incorrect idx - norm': wandb.Image(plot_wrong_norm(wrong_dict))})

    cfm_all = plot_confusion_matrices(wrong_dict)
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
        p = ast.literal_eval(p)
        t = ast.literal_eval(t)
        if len(p[:p.index("<PAD>")]) if "<PAD>" in p else len(p) == len(t[:t.index("<PAD>")]) if "<PAD>" in t else len(t):
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
    amount_in_one_pic = 6
    # all_pic = math.ceil(len(wrong)/6) # each picture has 6 images
    # for idx_pic in range(math.ceil(len(wrong)/6)):
    #     rows = math.ceil(idx_pic*6/2)
    #     cols = 2
    # fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 24))
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3, hspace=0.5)

    all_true = []; all_pred = []

    for index, data in wrong.items():  # Iterate over dictionary entries
        if index%amount_in_one_pic == 0:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 24))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3, hspace=0.5)
        true_labels, pred_labels = data["true"], data["pred"]  # Extract labels
        all_true.extend(true_labels); all_pred.extend(pred_labels)
        print(index-(amount_in_one_pic*(index//amount_in_one_pic)))
        row, col = divmod((index-(amount_in_one_pic*(index//amount_in_one_pic))), 2)  # Calculate row and col for subplot placement
        print("row", row, " col", col)
        ConfusionMatrixDisplay.from_predictions(true_labels, pred_labels, cmap=plt.cm.Blues, ax=axes[row, col], labels=["whole", "half", "quarter", "8th", "16th", "<EOS>", "<PAD>"])
        axes[row, col].set_title(f"Confusion Matrix at index {index}")
        if index%amount_in_one_pic == 5 or index == len(wrong)-1:
            print(divmod(index, amount_in_one_pic) )
            plt.close()
            wandb.log({f'Confusion Matrix/Index{index-(row*2 + col)}-{index}': wandb.Image(fig)})

    fig_all, ax = plt.subplots(1, 1, figsize=(8,10))
    ConfusionMatrixDisplay.from_predictions(all_true, all_pred, cmap=plt.cm.Blues, ax=ax, labels=["whole", "half", "quarter", "8th", "16th", "<EOS>", "<PAD>"])
    plt.title("Confusion Matrix")
    plt.close()

    return fig_all



if __name__ == "__main__":


    LOG_PATH = "09_chunks/log/dataset/long_length_data/transformer_conv2d_dataset/long_length_data_120240928-1718"
    output_dict = None

    wandb.init(entity='rhythm-tempo-project', project='chunks_train', id='w1vgzj1e', resume='must')

    print(InferencePlot(LOG_PATH, output_dict, wandb))



    
    print("completed")
