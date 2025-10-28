from srcs.data import (
    load_csv,
    display_data,
    discrimination_score,
    split_dataset
)
from srcs.share import (
    hist_col,
    box_col,
    bi_box_col,
    confusion_matrix,
    scatterplot_matrix,
    kdeplot
)


def main():
    try:
        datas = load_csv()
        # display_data(datas)
        # hist_col(datas)
        # box_col(datas)
        # bi_box_col(datas)
        # confusion_matrix(datas)
        # scatterplot_matrix(datas)
        # kdeplot(datas, datas.columns[:2], hue="diagnosis")
        # discrimination_score(datas)
        split_dataset(datas)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
