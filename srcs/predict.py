import json
import numpy as np
from srcs.Model import Model
from srcs.data import min_max_n
from configs.config import FEATURES, MODEL_SHAPE


def retrieve_min_max() -> list:
    with open("./results/utils/data_information.json") as file:
        min_max_training_data = json.loads(file.read())
    return min_max_training_data


def preprocess_data(min_max_training_data) -> np.ndarray:
    inputs = {}
    for ft in FEATURES:
        inputs[ft] = float(input(f"Enter a value for {ft}: "))
    features_min = min_max_training_data[0]
    features_max = min_max_training_data[1]
    normalized_inputs = []
    for ft in inputs.keys():
        normalized_inputs.append([min_max_n(
            min_v=features_min[ft],
            max_v=features_max[ft],
            value=inputs[ft]
        )])
    return np.array(normalized_inputs)


def predict():
    try:
        choice = input("Load last checkpoint ? y/n: ")
        checkpoint = True if choice == "y" else False
        model = Model(
            layers_shape=MODEL_SHAPE,
            load_last_checkpoint=checkpoint
        )
        min_max_training_data = retrieve_min_max()
        while True:
            print("\n====================================")
            model_inputs = preprocess_data(min_max_training_data)
            answer = model.predict(model_inputs)
            print(f"Results: {answer}")
            print("====================================\n")
    except KeyboardInterrupt:
        print("\nBye bye !")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    predict()
