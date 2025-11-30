import argparse
from pathlib import Path
from typing import Optional

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


def load_cell_types(cell_types_path: Path, mask_label: Optional[str] = None):
    cell_df = pd.read_csv(cell_types_path)
    cell_types = cell_df["cell_type"].tolist()
    if mask_label:
        cell_types = [f"{mask_label}::{ct}" for ct in cell_types]
    idx = cell_df["idx"].to_numpy() if "idx" in cell_df.columns else None
    return cell_types, idx


def load_reproduction_matrix(csv_path: Path, cell_types: list) -> np.ndarray:
    df = pd.read_csv(csv_path)

    num_features = len(FEATURE_ORDER)
    num_cells = len(cell_types)
    matrix = np.zeros((num_features, num_cells), dtype=np.float64)

    value_dict = dict(zip(df["feature"], df["value"]))

    for col_idx, cell in enumerate(cell_types):
        for row_idx, feature in enumerate(FEATURE_ORDER):
            key = f"{feature}_{cell}"
            if key not in value_dict:
                raise KeyError(f"Missing feature '{key}' in {csv_path}")
            matrix[row_idx, col_idx] = value_dict[key]

    return matrix


def compare_datasets(official: np.ndarray, reproduction: np.ndarray):
    if official.shape != reproduction.shape:
        raise ValueError(
            f"Shape mismatch: official {official.shape}, reproduction {reproduction.shape}"
        )

    diff = np.abs(official - reproduction)
    print("Comparison between official feature_dataset.npy and reproduction:")
    print(f"  shape: {official.shape}")
    print(f"  max abs diff: {diff.max():.6e}")
    print(f"  mean abs diff: {diff.mean():.6e}")
    print(f"  median abs diff: {np.median(diff):.6e}")
    print(f"  99th percentile abs diff: {np.percentile(diff, 99):.6e}")


def main():
    parser = argparse.ArgumentParser(description="Compare feature datasets.")
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("lionheart_reference_features/feature_dataset.npy"),
        help="Path to official feature_dataset.npy",
    )
    parser.add_argument(
        "--reproduction_csv",
        type=Path,
        default=Path("features_all_22chroms_10bp.csv"),
        help="Path to reproduction CSV output",
    )
    parser.add_argument(
        "--cell_types",
        type=Path,
        nargs="+",
        default=[
            Path("data/lionheart_resources/ATAC.idx_to_cell_type.csv"),
            Path("data/lionheart_resources/DNase.idx_to_cell_type.csv"),
        ],
        help="One or more idx_to_cell_type.csv files (order matters, default ATAC then DNase)",
    )
    parser.add_argument(
        "--mask_labels",
        type=str,
        nargs="*",
        help="Optional labels matching --cell_types to disambiguate mask types (e.g., DNase ATAC). Defaults to filename prefix.",
    )
    parser.add_argument(
        "--output_npy",
        type=Path,
        default=Path("features_all_22chroms_10bp.npy"),
        help="Optional path to save reproduction matrix",
    )
    args = parser.parse_args()

    combined_cell_types = []
    combined_indices = []
    mask_labels = args.mask_labels or []
    if mask_labels and len(mask_labels) != len(args.cell_types):
        raise ValueError("--mask_labels must match the number of --cell_types entries")

    offset = 0
    for i, path in enumerate(args.cell_types):
        label = (
            mask_labels[i]
            if mask_labels
            else path.stem.split(".")[0]
        )
        cell_df = pd.read_csv(path)
        idx_values = (
            cell_df["idx"].to_numpy()
            if "idx" in cell_df.columns
            else np.arange(len(cell_df))
        )
        combined_cell_types.extend(f"{label}::{ct}" for ct in cell_df["cell_type"])
        combined_indices.extend((offset + idx_values).tolist())
        offset += len(cell_df)

    official_all = np.load(args.official)
    official = official_all[:, combined_indices]

    reproduction = load_reproduction_matrix(
        args.reproduction_csv, combined_cell_types
    )

    if args.output_npy:
        np.save(args.output_npy, reproduction)
        print(f"Saved reproduction matrix to {args.output_npy}")

    compare_datasets(official, reproduction)


if __name__ == "__main__":
    main()

