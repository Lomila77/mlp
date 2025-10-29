from srcs.Model import Model
from srcs.data import data, load_data
from srcs.share import (
    plot_loss,
    save_training_metrics,
    save_min_max_training_data
)
from configs.config import (
    FEATURES,
    CATEGORIES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SHAPE
)
import numpy as np
import pandas as pd


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    train_dataset: tuple = load_data(
        train_df, BATCH_SIZE, FEATURES, CATEGORIES)
    val_dataset: tuple = load_data(
        val_df, BATCH_SIZE, FEATURES, CATEGORIES)
    return train_dataset, val_dataset


def save_training(
    train_df: pd.DataFrame, model: Model, metrics: dict
):
    min_features = {}
    max_features = {}
    for ft in FEATURES:
        min_features[ft] = train_df[ft].min()
        max_features[ft] = train_df[ft].max()
    save_min_max_training_data(min_features, max_features)
    metrics["models_shape"] = MODEL_SHAPE
    metrics["epochs"] = EPOCHS
    metrics["learning_rate"] = LEARNING_RATE
    plot_loss(metrics["loss"])
    plot_loss(metrics["v_loss"], name="Validation loss")
    print("====================================")
    print("Results:")
    print(f"Accuracy: {(metrics['accuracy'] * 100):.2f} %")
    print(f"Precision: {(metrics['precision'] * 100):.2f} %")
    print(f"Recall: {(metrics['recall'] * 100):.2f} %")
    print("====================================")
    last_loss = metrics["loss"][-1]
    last_v_loss = metrics["v_loss"][-1]
    del metrics["loss"]
    del metrics["v_loss"]
    metrics["loss"] = last_loss
    metrics["v_loss"] = last_v_loss
    save_training_metrics(metrics)
    model.save_model()


def train():
    try:
        train_df, val_df = data()
        train_dataset, val_dataset = preprocess_data(
            train_df, val_df
        )
        print("\n\n====================================")
        model = Model(MODEL_SHAPE, BATCH_SIZE, LEARNING_RATE)
        print(f"Training on: {', '.join(FEATURES)}")
        print("====================================\n\n")
        metrics = model.train(train_dataset, val_dataset, EPOCHS)
        save_training(train_df, model, metrics)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    train()
