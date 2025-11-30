import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Tuple

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


def load_feature_matrix(csv_path: Path, drop_consensus: bool = False) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    df = pd.read_csv(csv_path)
    values = dict(zip(df["feature"], df["value"]))
    mask_cells: List[Tuple[str, str]] = []

    for feat in df["feature"]:
        prefix = next((fo for fo in FEATURE_ORDER if feat.startswith(f"{fo}_")), None)
        if prefix is None:
            raise ValueError(f"Unrecognized feature prefix in {feat}")
        rest = feat[len(prefix) + 1 :]
        if "::" in rest:
            mask, cell = rest.split("::", 1)
        else:
            mask, cell = "DNase", rest
        key = (mask, cell)
        if drop_consensus and cell == "consensus":
            continue
        if key not in mask_cells:
            mask_cells.append(key)

    mat = np.zeros((len(FEATURE_ORDER), len(mask_cells)), dtype=np.float64)
    for col, (mask, cell) in enumerate(mask_cells):
        for row, feat in enumerate(FEATURE_ORDER):
            key = f"{feat}_{mask}::{cell}"
            fallback = f"{feat}_{cell}"
            if key not in values and fallback in values:
                key = fallback
            if key not in values:
                raise KeyError(f"Missing feature {key}")
            mat[row, col] = values[key]

    return mat, mask_cells


def save_variants(matrix: np.ndarray, mask_cells: List[Tuple[str, str]], out_prefix: Path):
    order_atac = [idx for idx, (mask, _) in enumerate(mask_cells) if mask.upper() == "ATAC"]
    order_dnase = [idx for idx, (mask, _) in enumerate(mask_cells) if mask.upper() == "DNASE"]

    base = out_prefix.stem
    directory = out_prefix.parent
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / f"{base}_original_order.npy", matrix)
    np.save(
        directory / f"{base}_ATAC_DNase.npy",
        matrix[:, order_atac + order_dnase],
    )
    np.save(directory / f"{base}_DNase_ATAC.npy", matrix[:, order_dnase + order_atac])
    print(
        f"Saved matrices (base={base}): original={matrix.shape}, ATAC-first={len(order_atac)}, DNase-first={len(order_dnase)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorder reproduction feature matrix.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("reproduce/features_all_22chroms_10bp.csv"),
        help="Path to reproduction CSV",
    )
    parser.add_argument(
        "--drop_consensus",
        action="store_true",
        help="Exclude consensus columns from the matrix",
    )
    args = parser.parse_args()

    matrix, mask_cells = load_feature_matrix(args.csv, drop_consensus=args.drop_consensus)
    suffix = "_no_consensus" if args.drop_consensus else ""
    save_variants(matrix, mask_cells, Path(f"reproduce/features_matrix{suffix}.npy"))

