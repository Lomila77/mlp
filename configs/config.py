FEATURES = ["area_worst", "concavity_mean"]
CATEGORIES = ["M", "B"]

BATCH_SIZE = 3
EPOCHS = 30
LEARNING_RATE = 0.7
MODEL_SHAPE = [len(FEATURES), 3, len(CATEGORIES)]

RESULTS_PATH = "./results/"
ANALYSIS_PATH = "./analysis/graph/"
MODEL_PATH = RESULTS_PATH + "utils/model_weights.pkl"
