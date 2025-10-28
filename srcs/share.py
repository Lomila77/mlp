
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


RESULTS_PATH = "./results/"
ANALYSIS_PATH = "./analysis/graph/"


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


def plot_loss(loss_values: list[float], name: str = "Loss"):
    plt.figure(figsize=(12, 8))
    df = pd.DataFrame({
        "step": list(range(1, len(loss_values) + 1)),
        "loss": loss_values
    })
    sns.lineplot(data=df, x="step", y="loss")
    plt.title(name)
    plt.xlabel("Step")
    plt.ylabel(name)
    save_fig(name, RESULTS_PATH + "graph")
    plt.close()


def save_training(
    models_shape: list[int],
    epochs: int,
    learning_rate: float,
    loss: list[float],
    validation_loss: list[float],
    accuracy: float,
    precision: float,
    recall: float
) -> None:
    file_path = RESULTS_PATH + 'score/training_metrics.csv'
    new_entry = {
        'models_shape': str(models_shape),
        'epochs': epochs,
        'learning_rate': learning_rate,
        'loss': np.mean(loss),
        'validation_loss': np.mean(validation_loss),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        next_id = df['id'].max() + 1 if not df.empty else 1
    else:
        df = pd.DataFrame()
        next_id = 1

    new_entry['id'] = next_id
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    # Mettre les colonnes dans l'ordre
    df = df[
        [
            'id',
            'models_shape',
            'epochs',
            'learning_rate',
            'loss',
            'validation_loss',
            'accuracy',
            'precision',
            'recall']]
    df.to_csv(file_path, index=False)


def save_fig(name: str, folder: str):
    """Save fig at folder given path, with the given name"""
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{name}.png")
