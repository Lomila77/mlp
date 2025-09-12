import pandas as pd


def load_csv(path: str = "./data.csv") -> pd.DataFrame:
    cols = [
        "id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    return pd.read_csv(path, names=cols, header=None)


def display_data(datas: pd.DataFrame):
    datas.head()
    datas.info()
    datas.describe()


def split_dataset():
    pass


def data():
    pass


if __name__ == "main":
    data()
