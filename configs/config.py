FEATURES = ["area_worst", "concavity_mean", "perimeter_worst", "radius_worst"]
CATEGORIES = ["M", "B"]

BATCH_SIZE = 6
EPOCHS = 8
LEARNING_RATE = 0.7
MODEL_SHAPE = [len(FEATURES), 4, len(CATEGORIES)]

RESULTS_PATH = "./results/"
ANALYSIS_PATH = "./analysis/graph/"
MODEL_PATH = RESULTS_PATH + "utils/model_weights.pkl"

TRAIN_DATASET_PATH = "data_training.csv"
VAL_DATASET_PATH = "data_test.csv"
