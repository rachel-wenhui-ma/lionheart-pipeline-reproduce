import argparse
from pathlib import Path

from collections import Counter

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


def build_matrix(csv_path: Path, mask_label: str, csv: Path):
    df = pd.read_csv(csv_path)
    cell_types = [f"{mask_label}::{ct}" for ct in df["cell_type"]]
    values = pd.read_csv(csv).set_index("feature")["value"].to_dict()

    matrix = np.zeros((len(FEATURE_ORDER), len(cell_types)), dtype=np.float64)
    for col, cell in enumerate(cell_types):
        for row, feat in enumerate(FEATURE_ORDER):
            key = f"{feat}_{cell}"
            if key not in values:
                raise KeyError(f"Missing feature '{key}' in {csv}")
            matrix[row, col] = values[key]
    return matrix, cell_types, df["idx"].to_numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Compare DNase-only reproduction features to the official matrix."
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("reproduce/lionheart_reference_features/feature_dataset.npy"),
    )
    parser.add_argument(
        "--reproduction_csv",
        type=Path,
        default=Path("reproduce/features_all_22chroms_10bp.csv"),
    )
    parser.add_argument(
        "--dnase_idx",
        type=Path,
        default=Path("reproduce/data/lionheart_resources/DNase.idx_to_cell_type.csv"),
    )
    parser.add_argument(
        "--atac_idx",
        type=Path,
        default=Path("reproduce/data/lionheart_resources/ATAC.idx_to_cell_type.csv"),
    )
    parser.add_argument("--precision", type=int, default=6)
    args = parser.parse_args()

    official = np.load(args.official)
    dnase_matrix, cell_names, idx_vals = build_matrix(
        args.dnase_idx, "DNase", args.reproduction_csv
    )

    atac_count = len(pd.read_csv(args.atac_idx))
    dnase_indices = atac_count + idx_vals

    official_dnase = official[:, dnase_indices]
    if official_dnase.shape != dnase_matrix.shape:
        raise ValueError(
            f"Shape mismatch: official {official_dnase.shape}, "
            f"reproduction {dnase_matrix.shape}"
        )

    diff = official_dnase - dnase_matrix

    for feat in ("x_sum", "x_squared_sum"):
        row = FEATURE_ORDER.index(feat)
        per_cell = diff[row]
        rounded = np.round(per_cell, args.precision)
        counts = Counter(rounded)
        unique = np.array(sorted(counts.keys()))
        print(f"{feat} diff stats:")
        print(f"  max abs diff: {np.max(np.abs(per_cell)):.6g}")
        print(f"  mean abs diff: {np.mean(np.abs(per_cell)):.6g}")
        print(
            f"  unique rounded diffs ({len(unique)}): "
            f"{', '.join(map(str, unique[:10]))}"
        )
        print("  counts:")
        for value in unique:
            print(f"    diff={value:.{args.precision}f} count={counts[value]}")
            sample_idx = np.where(rounded == value)[0][:3]
            for idx in sample_idx:
                print(f"      e.g., {cell_names[idx]}")
        worst = np.argsort(-np.abs(per_cell))[:5]
        for idx in worst:
            print(f"    {cell_names[idx]} diff={per_cell[idx]:.{args.precision}f}")
        print()

    np.save(
        args.reproduction_csv.with_suffix(".dnase_diff.npy"),
        diff,
    )
    print(
        f"Saved per-feature diff matrix to "
        f"{args.reproduction_csv.with_suffix('.dnase_diff.npy')}"
    )


if __name__ == "__main__":
    main()

