import pandas as pd
import numpy as np
import os
from configs.config import TRAIN_DATASET_PATH, VAL_DATASET_PATH


def load_csv(path: str = "./data.csv") -> pd.DataFrame:
    """Load the CSV and put some columns name"""
    cols = [
        "id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
        "fractal_dimension_se", "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ]
    return pd.read_csv(path, names=cols, header=None)

def one_hot_labels_encoding(
    labels: np.ndarray, categories: list[str]
) -> np.ndarray:
    """Encode labels for the model
    0 = "M" | 1 = "B"
    """
    encoded_labels = np.zeros((len(categories), len(labels)))
    for idx, lab in enumerate(labels):
        g_t = categories.index(lab)
        encoded_labels[g_t][idx] = 1
    return encoded_labels


def min_max_n(min_v: float, max_v: float, value: float) -> float:
    """Normalize one input for the given min / max.

    Args:
        min_v (float): min value in the scale
        max_v (float): max value in the scale
        value (float): the value to normalize

    Returns:
        float: The value normalized
    """
    norm_range = max_v - min_v
    if norm_range == 0:
        return 0
    return (value - min_v) / norm_range


def min_max_inputs_normalization(inputs: np.ndarray) -> np.ndarray:
    """Normalize inputs for model ingestion.
    Row are features input, columns are the number of example.

    Args:
        inputs (np.ndarray): Input to ingest

    Returns:
        np.ndarray: Normalized input
    """
    normalized_inputs = np.zeros_like(inputs)
    for idx, feature in enumerate(inputs):
        min_val = feature.min()
        max_val = feature.max()
        norm_range = max_val - min_val
        if norm_range == 0:
            normalized_inputs[idx] = np.zeros_like(feature)
        else:
            normalized_inputs[idx] = (feature - min_val) / norm_range
    return normalized_inputs


def load_data(
    dataset: pd.DataFrame,
    batch_size: int,
    in_features: list[str],
    out_categories: list[str]
) -> list[tuple]:
    """
    Loads and preprocesses the dataset for model training.
    """
    # Selection de la feature area_worst donc modele a une seule entree
    inputs = dataset[in_features].to_numpy().T
    labels = dataset["diagnosis"].to_numpy()
    encoded_labels = one_hot_labels_encoding(labels, out_categories)
    normalized_inputs = min_max_inputs_normalization(inputs)

    n_samples = normalized_inputs.shape[1]  # Nombre de colonnes = nombre d'échantillons
    batches: list[tuple] = []
    for i in range(0, n_samples, batch_size):
        batch_inputs = normalized_inputs[:, i:i+batch_size]
        batch_labels = encoded_labels[:, i:i+batch_size]
        assert len(batch_inputs[0]) == len(batch_labels[0])
        batches.append((batch_inputs, batch_labels))
    return batches


def discrimination_score(datas: pd.DataFrame):
    folder_path = "./analysis/graph/score"
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    scores = {
        "columns": [],
        "scores": [],
        "max_b": [],
        "min_b": [],
        "max_m": [],
        "min_m": []
    }
    for col in datas.columns:
        if col == "id" or col == "diagnosis":
            continue
        m_group = datas[datas["diagnosis"] == "M"][col]
        b_group = datas[datas["diagnosis"] == "B"][col]
        diff = b_group.max() - m_group.min()
        norm_range = datas[col].max() - datas[col].min()
        scores["columns"].append(col)
        scores["scores"].append(
            diff / norm_range if norm_range != 0 else 0)
        scores["max_b"].append(b_group.max())
        scores["min_b"].append(b_group.min())
        scores["max_m"].append(m_group.max())
        scores["min_m"].append(m_group.min())
        pd.DataFrame(scores).to_csv(
            f"{folder_path}/discrimination.csv",
            index=False
        )


def display_data(datas: pd.DataFrame):
    """Display basic information about datas"""
    datas.head()
    datas.info()
    datas.describe()


def split_dataset(datas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    datas.sample(frac=1)
    slice_idx = int(0.7 * len(datas))
    train_dataset = datas[:slice_idx]
    validation_dataset = datas[slice_idx:]
    return train_dataset, validation_dataset


def data():
    try:
        datas = load_csv()
        df_train, df_val = split_dataset(datas)
        print("==============================================================")
        print("Train set:")
        display_data(df_train)
        print("==============================================================")
        print("Val set:")
        display_data(df_val)
        print("==============================================================")
        df_train.to_csv(TRAIN_DATASET_PATH)
        df_val.to_csv(VAL_DATASET_PATH)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    data()
