FEATURES = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean",
        "radius_se", "perimeter_se", "area_se",
        "concave_points_se",
        "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ]
CATEGORIES = ["M", "B"]

BATCH_SIZE = 4
EPOCHS = 40000
LEARNING_RATE = 0.0001
MODEL_SHAPE = [len(FEATURES), 64, 128, 64, len(CATEGORIES)]

RESULTS_PATH = "./results/"
ANALYSIS_PATH = "./analysis/graph/"
MODEL_PATH = RESULTS_PATH + "utils/model_weights.pkl"

TRAIN_DATASET_PATH = "data_training.csv"
VAL_DATASET_PATH = "data_test.csv"
