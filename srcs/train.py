from srcs.Model import Model
from srcs.data import (
    data,
    load_data
)
import numpy as np


def train():
    # try:
        out_categories: list = ["M", "B"]
        in_features: list = ["area_worst"]
        shape: list = [len(in_features), 3, 3, len(out_categories)]
        batch_size = 3

        train_df, val_df = data()
        train_dataset: tuple = load_data(
            train_df, batch_size, in_features, out_categories)
        val_dataset: tuple = load_data(
            val_df, batch_size, in_features, out_categories)
        model = Model(shape, batch_size)
        for input, label in train_dataset:
            print(label.shape)
            print(type(label))
            model.train(input, label)
        # model = Model(shape, input_model, label)
        # print(model)
    # except Exception as e:
    #     print(e)


if __name__ == "__main__":
    train()
