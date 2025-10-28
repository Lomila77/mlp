from srcs.Model import Model
from srcs.data import (
    data,
    load_data
)
from srcs.share import plot_loss


def train():
    try:
        batch_size = 3
        epochs = 80
        learning_rate = 0.7

        out_categories: list = ["M", "B"]
        in_features: list = ["area_worst"]
        train_df, val_df = data()
        train_dataset: tuple = load_data(
            train_df, batch_size, in_features, out_categories)
        val_dataset: tuple = load_data(
            val_df, batch_size, in_features, out_categories)
        print(f"Len dataset: {len(train_dataset)}")

        shape: list = [len(in_features), 3, 3, len(out_categories)]
        model = Model(shape, batch_size, learning_rate)
    
        metrics = model.train(train_dataset, val_dataset, epochs)
        plot_loss(metrics["loss"])
        plot_loss(metrics["v_loss"], name="v_losses")
        print(f"Accuracy: {(metrics['accuracy'] * 100):.2f} %")
        print(f"Precision: {(metrics['precision'] * 100):.2f} %")
        print(f"Recall: {(metrics['recall'] * 100):.2f} %")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    train()
