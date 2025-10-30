
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from configs.config import RESULTS_PATH, ANALYSIS_PATH


def hist_col(datas: pd.DataFrame):
    for col in datas.columns:
        plt.figure(figsize=(12, 8))
        datas[col].hist()
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Value")
        save_fig(col, ANALYSIS_PATH + "Histogramme")


def box_col(datas: pd.DataFrame):
    for col in datas.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=datas[col])
        plt.title(col)
        save_fig(col, ANALYSIS_PATH + "Simple_Boxplot")


def bi_box_col(datas: pd.DataFrame):
    for col in datas.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=datas, x="diagnosis", y=col)
        plt.title(f"Boxplot {col} with \"Value\"")
        plt.xlabel("M, B")
        plt.ylabel(col)
        save_fig(col, ANALYSIS_PATH + "Bi_Boxplot")
        plt.close()


def confusion_matrix(datas: pd.DataFrame):
    purge_datas = datas.drop(columns=["id", "diagnosis"])
    plt.figure(figsize=(20, 15))
    sns.heatmap(purge_datas.corr(), cmap='coolwarm', annot=True)
    plt.title('Heatmap de corrélation des variables numériques')
    save_fig("confusion", ANALYSIS_PATH + "Matrix")


def scatterplot_matrix(datas: pd.DataFrame):
    sns.set_theme(style="ticks")
    sns.pairplot(datas, hue="diagnosis")
    save_fig("scatterplot", ANALYSIS_PATH + "Matrix")


def kdeplot(datas: pd.DataFrame, columns: list[str], hue: str):
    sns.set_theme(style="ticks")
    for col in columns:
        sns.kdeplot(data=datas, x=col, hue=hue, fill=True)
        save_fig(f"{col}", ANALYSIS_PATH + "KDEplot")
        plt.close()


def multi_line_plot(
    df: pd.DataFrame,
    name: str = "Loss"
):
    sns.set_theme(style="darkgrid")
    sns.set_context(context="paper")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x="step", y="value", hue="type", marker='o')
    ax.set_title(name)
    ax.set_xlabel("Step")
    ax.set_ylabel(name)
    ax.set_ylim(0, 1)
    plt.legend(title="Courbe")
    plt.tight_layout()
    save_fig(name, RESULTS_PATH + "graph")
    plt.close()


def plot_loss(
    loss_values: list[float],
    v_losses_values: list[float],
):
    if len(loss_values) > len(v_losses_values):
        loss_values = loss_values[:len(v_losses_values)]
    steps = list(range(1, len(loss_values) + 1))
    df = pd.DataFrame({
        "step": steps + steps,
        "value": loss_values + v_losses_values,
        "type": ["Loss"] * len(loss_values) +
                ["Validation loss"] * len(v_losses_values)
    })
    multi_line_plot(df)


def plot_metrics(metrics: dict):

    accuracy: list[float] = metrics["accuracy"]
    precisiosn: list[float] = metrics["precision"]
    recall: list[float] = metrics["recall"]
    f1: list[float] = metrics["f1"]
    steps = list(range(1, len(accuracy) + 1))
    df = pd.DataFrame({
        "step": steps * 4,
        "value": accuracy + precisiosn + recall + f1,
        "type": ["Accuracy"] * len(accuracy) +
                ["Precision"] * len(precisiosn) +
                ["Recall"] * len(recall) +
                ["F1"] * len(f1)
    })
    multi_line_plot(df, name="Metrics")


def save_training_results(results: dict) -> None:
    file_path = RESULTS_PATH + 'score/training_results.csv'
    for key, val in results.items():
        if isinstance(val, (np.floating, float)):
            results[key] = f"{float(val):.2f}"
        elif isinstance(val, (np.ndarray, list)):
            results[key] = str(list(val))
    df = pd.DataFrame([results])
    if os.path.exists(file_path):
        next_id = pd.read_csv(
            file_path)['id'].max() + 1 if not df.empty else 1
        csv_kwargs = {'mode': 'a', 'header': False, 'index': False}
    else:
        next_id = 1
        csv_kwargs = {'index': False}
    df.insert(0, 'id', next_id)
    df.to_csv(file_path, **csv_kwargs)


def save_min_max_training_data(
    min_features: dict,
    max_features: dict
):
    file_path = RESULTS_PATH + "utils/data_information.json"
    with open(file_path, "+w") as f:
        f.write(json.dumps([min_features, max_features]))


def save_fig(name: str, folder: str):
    """Save fig at folder given path, with the given name"""
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{name}.png")
