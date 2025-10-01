
from matplotlib import axes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def hist_col(datas: pd.DataFrame):
    for col in datas.columns:
        plt.figure(figsize=(12, 8))
        datas[col].hist()
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Value")
        save_fig(col, "Histogramme")


def box_col(datas: pd.DataFrame):
    for col in datas.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=datas[col])
        plt.title(col)
        save_fig(col, "Simple_Boxplot")


def bi_box_col(datas: pd.DataFrame):
    for col in datas.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=datas, x="diagnosis", y=col)
        plt.title(f"Boxplot {col} with \"Value\"")
        plt.xlabel("M, B")
        plt.ylabel(col)
        save_fig(col, "Bi_Boxplot")
        plt.close()


def confusion_matrix(datas: pd.DataFrame):
    purge_datas = datas.drop(columns=["id", "diagnosis"])
    plt.figure(figsize=(20, 15))
    sns.heatmap(purge_datas.corr(), cmap='coolwarm', annot=True)
    plt.title('Heatmap de corrélation des variables numériques')
    save_fig("confusion", "Matrix")


def scatterplot_matrix(datas: pd.DataFrame):
    sns.set_theme(style="ticks")
    sns.pairplot(datas, hue="diagnosis")
    save_fig("scatterplot", "Matrix")


def kdeplot(datas: pd.DataFrame):
    sns.set_theme(style="ticks")
    data_sliced = datas.columns[2:]
    for col in data_sliced:
        sns.kdeplot(data=datas, x=col, hue="diagnosis", fill=True)
        save_fig(f"{col}", "KDEplot")
        plt.close()

def save_fig(name: str, folder: str):
    """Save fig at folder given path, with the given name"""
    folder_path = f"./analysis/graph/{folder}"
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    plt.savefig(f"{folder_path}/{name}.png")
