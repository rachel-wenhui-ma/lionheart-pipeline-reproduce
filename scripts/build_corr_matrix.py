import numpy as np
import pandas as pd

FEATURE_ORDER = [
    "pearson_r",
    "p_value",
    "fraction_within",
    "cosine_similarity",
    "x_sum",
    "y_sum",
    "x_squared_sum",
    "y_squared_sum",
    "xy_sum",
    "n",
]


def main():
    corr = np.load(
        "features_all_22chroms_10bp.corr_stats.npy", allow_pickle=True
    ).item()
    atac = pd.read_csv(
        "reproduce/data/lionheart_resources/ATAC.idx_to_cell_type.csv"
    )["cell_type"].tolist()
    dnase = pd.read_csv(
        "reproduce/data/lionheart_resources/DNase.idx_to_cell_type.csv"
    )["cell_type"].tolist()
    cell_names = atac + dnase
    mask_labels = ["ATAC"] * len(atac) + ["DNase"] * len(dnase)

    mat = np.zeros((len(FEATURE_ORDER), len(cell_names)), dtype=np.float64)
    for col, (mask, cell) in enumerate(zip(mask_labels, cell_names)):
        prefix = f"{mask}::{cell}"
        for row, feat in enumerate(FEATURE_ORDER):
            key = f"{feat}_{prefix}"
            mat[row, col] = corr.get(key, 0.0)

    np.save("reproduce/features_matrix_corr_stats.npy", mat)
    print("saved matrix", mat.shape)


if __name__ == "__main__":
    main()




