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


def load_cell_order(resources_dir: Path):
    atac = pd.read_csv(resources_dir / "ATAC.idx_to_cell_type.csv")["cell_type"].tolist()
    dnase = pd.read_csv(resources_dir / "DNase.idx_to_cell_type.csv")["cell_type"].tolist()
    return atac + dnase, len(atac)


def main():
    parser = argparse.ArgumentParser(description="Compare reproduction vs official columns.")
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["wtc11__cl"],
        help="Cell types to inspect (use raw name, not prefixed)",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("reproduce/features_matrix_ATAC_DNase.npy"),
        help="Path to reproduction matrix",
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("reproduce/lionheart_reference_features/feature_dataset.npy"),
        help="Path to official feature_dataset.npy",
    )
    parser.add_argument(
        "--resources",
        type=Path,
        default=Path("reproduce/data/lionheart_resources"),
        help="Path to local resources dir with idx CSVs",
    )
    args = parser.parse_args()

    repro = np.load(args.matrix)
    off = np.load(args.official)
    cell_order, atac_len = load_cell_order(args.resources)

    for cell in args.cells:
        requested_mask = None
        name = cell
        if "::" in cell:
            requested_mask, name = cell.split("::", 1)
        candidates = [i for i, ct in enumerate(cell_order) if ct == name]
        if requested_mask:
            requested_mask = requested_mask.upper()
            candidates = [
                i
                for i in candidates
                if (i < atac_len and requested_mask == "ATAC")
                or (i >= atac_len and requested_mask == "DNASE")
            ]
        if not candidates:
            print(f"Cell {cell} not found")
            continue
        idx = candidates[0]
        mask = "ATAC" if idx < atac_len else "DNase"
        print(f"\n=== {mask}::{name} (column {idx}) ===")
        for row, feat in enumerate(FEATURE_ORDER):
            rv = repro[row, idx]
            ov = off[row, idx]
            print(f"{feat:<20} repro={rv:>12.6g}  official={ov:>12.6g}  diff={abs(rv-ov):>12.6g}")


if __name__ == "__main__":
    main()

