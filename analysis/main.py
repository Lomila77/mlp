from srcs.data import load_csv, display_data
from srcs.share import (
    hist_col,
    box_col,
    bi_box_col,
    confusion_matrix
)


def main():
    try:
        datas = load_csv()
        display_data(datas)
        hist_col(datas)
        box_col(datas)
        bi_box_col(datas)
        confusion_matrix(datas)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
