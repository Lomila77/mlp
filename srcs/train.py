from srcs.Model import Model
from srcs.data import load_csv, load_data
from srcs.share import (
    plot_loss,
    plot_metrics,
    save_training_results,
    save_min_max_training_data
)
from configs.config import (
    FEATURES,
    CATEGORIES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SHAPE,
    TRAIN_DATASET_PATH,
    VAL_DATASET_PATH
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
    results = {}
    min_features = {}
    max_features = {}
    for ft in FEATURES:
        min_features[ft] = train_df[ft].min()
        max_features[ft] = train_df[ft].max()
    save_min_max_training_data(min_features, max_features)
    results["epochs"] = metrics["epochs"]
    results["models_shape"] = MODEL_SHAPE
    results["learning_rate"] = LEARNING_RATE
    results["batch_size"] = BATCH_SIZE
    plot_loss(metrics["loss"], metrics["v_loss"])
    results["loss"] = metrics["loss"][-1]
    results["v_loss"] = metrics["v_loss"][-1]
    plot_metrics(metrics)
    results["accuracy"] = np.mean(metrics['accuracy'])
    results["precision"] = np.mean(metrics['precision'])
    results["recall"] = np.mean(metrics['recall'])
    results["f1"] = np.mean(metrics['f1'])

    print("====================================")
    print("Results:")
    print(f"Accuracy: {(results['accuracy'] * 100):.2f} %")
    print(f"Precision: {(results['precision'] * 100):.2f} %")
    print(f"Recall: {(results['recall'] * 100):.2f} %")
    print(f"f1: {(results['f1'] * 100):.2f} %")
    print("====================================")
    save_training_results(results)
    model.save_model()


def train():
    try:
        train_df = load_csv(TRAIN_DATASET_PATH)
        val_df = load_csv(VAL_DATASET_PATH)
        train_dataset, val_dataset = preprocess_data(
            train_df, val_df
        )
        print("\n\n====================================")
        model = Model(MODEL_SHAPE, BATCH_SIZE, LEARNING_RATE)
        print(f"Training on: {', '.join(FEATURES)}")
        print("====================================\n\n")
        results = model.train(train_dataset, val_dataset, EPOCHS)
        save_training(train_df, model, results)
    except KeyboardInterrupt:
        print("Early stopping kill during saving results, files be missing")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    train()
