import argparse
from collections import Counter, defaultdict
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


def load_cell_types(csv_path: Path):
    df = pd.read_csv(csv_path)
    return df["cell_type"].tolist()


def summarize(diffs: np.ndarray, label: str, precision: int = 6):
    rounded = np.round(diffs, precision)
    counts = Counter(rounded)
    print(f"\n{label}:")
    for value, count in sorted(counts.items()):
        print(f"  diff={value:.{precision}f}  count={count}")


def main():
    parser = argparse.ArgumentParser(description="Trace x_sum/x_squared_sum offsets.")
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("reproduce/lionheart_reference_features/feature_dataset.npy"),
    )
    parser.add_argument(
        "--repro",
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
    parser.add_argument("--precision", type=int, default=6)
    args = parser.parse_args()

    official = np.load(args.official)
    repro = np.load(args.repro)

    if official.shape != repro.shape:
        raise ValueError(
            f"Shape mismatch: official {official.shape} vs reproduction {repro.shape}"
        )

    atac_cells = load_cell_types(args.atac_idx)
    dnase_cells = load_cell_types(args.dnase_idx)
    mask_cells = [("ATAC", ct) for ct in atac_cells] + [
        ("DNase", ct) for ct in dnase_cells
    ]
    if len(mask_cells) != official.shape[1]:
        raise ValueError(
            f"Cell list length {len(mask_cells)} does not match matrix columns {official.shape[1]}"
        )

    diff_x = repro[FEATURE_ORDER.index("x_sum")] - official[
        FEATURE_ORDER.index("x_sum")
    ]
    diff_xsq = repro[FEATURE_ORDER.index("x_squared_sum")] - official[
        FEATURE_ORDER.index("x_squared_sum")
    ]

    print("Overall offsets:")
    summarize(diff_x, "x_sum (all)", args.precision)
    summarize(diff_xsq, "x_squared_sum (all)", args.precision)

    grouped_x = defaultdict(list)
    grouped_xsq = defaultdict(list)
    for idx, (mask, cell) in enumerate(mask_cells):
        grouped_x[mask].append(diff_x[idx])
        grouped_xsq[mask].append(diff_xsq[idx])

    for mask in ("ATAC", "DNase"):
        if grouped_x[mask]:
            summarize(np.array(grouped_x[mask]), f"x_sum ({mask})", args.precision)
            summarize(
                np.array(grouped_xsq[mask]), f"x_squared_sum ({mask})", args.precision
            )

    for mask in ("ATAC", "DNase"):
        try:
            idx = (
                atac_cells.index("consensus")
                if mask == "ATAC"
                else len(atac_cells) + dnase_cells.index("consensus")
            )
        except ValueError:
            continue
        print(
            f"\n{mask}::consensus diffs -> "
            f"x_sum={diff_x[idx]:.{args.precision}f}, "
            f"x_squared_sum={diff_xsq[idx]:.{args.precision}f}"
        )


if __name__ == "__main__":
    main()



