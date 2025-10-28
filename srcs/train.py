from numpy import mean
from srcs.Model import Model
from srcs.data import (
    data,
    load_data
)
from srcs.share import plot_loss


def train():
    # try:
        batch_size = 3
        epochs = 180
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
        losses = []
        for epoch in range(epochs):
            loss = []
            print("====================================")
            print("TRAIN:")
            print(f"Epochs: {epoch}")
            for input, label in train_dataset:
                loss.append(model.train(input, label))
            # for input, label in val_dataset:
            #     prediction = model.predict(input)
            losses.append(mean(loss))
            plot_loss(losses)
            print("====================================")

    # except Exception as e:
    #     print(e)


if __name__ == "__main__":
    train()
