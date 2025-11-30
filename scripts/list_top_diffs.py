import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="List columns with largest differences")
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("reproduce/lionheart_reference_features/feature_dataset.npy"),
    )
    parser.add_argument(
        "--repro_matrix",
        type=Path,
        default=Path("reproduce/features_matrix_ATAC_DNase.npy"),
    )
    parser.add_argument(
        "--atac_idx",
        type=Path,
        default=Path("reproduce/data/lionheart_resources/ATAC.idx_to_cell_type.csv"),
    )
    parser.add_argument(
        "--dnase_idx",
        type=Path,
        default=Path("reproduce/data/lionheart_resources/DNase.idx_to_cell_type.csv"),
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    official = np.load(args.official)
    repro = np.load(args.repro_matrix)
    diff = np.abs(official - repro)

    atac_cells = pd.read_csv(args.atac_idx)["cell_type"].tolist()
    dnase_cells = pd.read_csv(args.dnase_idx)["cell_type"].tolist()
    cell_names = atac_cells + dnase_cells
    mask_labels = ["ATAC"] * len(atac_cells) + ["DNase"] * len(dnase_cells)

    col_max = diff.max(axis=0)
    sorted_idx = np.argsort(col_max)[::-1]

    print(f"Top {args.top} columns by max absolute difference:")
    for rank in range(min(args.top, len(sorted_idx))):
        idx = sorted_idx[rank]
        row = int(np.argmax(diff[:, idx]))
        print(
            f"{rank+1:02d} {mask_labels[idx]}::{cell_names[idx]:40s} "
            f"max_diff={col_max[idx]:>12.1f}  culprit={FEATURE_ORDER[row]} "
            f"(diff={diff[row, idx]:.1f})"
        )


if __name__ == "__main__":
    main()




