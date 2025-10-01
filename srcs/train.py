from srcs.Model import Model
from srcs.data import (
    data,
    load_data
)
import numpy as np


def train():
    # try:
        shape = [1, 3, 3, 2]
        train_df, val_df = data()
        train_dataset: tuple = load_data(train_df)
        val_dataset: tuple = load_data(val_df)
        print("Train dataset:")
        print(f"Inputs: {train_dataset[0].shape}")
        print(f"Labels: {train_dataset[1].shape}")
        print("Val dataset:")
        print(f"Inputs: {val_dataset[0].shape}")
        print(f"Labels: {val_dataset[1].shape}")
        for input, label in train_dataset:
            
        # model = Model(shape, input_model, label)
        # print(model)
    # except Exception as e:
    #     print(e)


if __name__ == "__main__":
    train()
