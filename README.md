# LIONHEART Feature Extraction Pipeline Reproduction

This repository contains a reproduction of LIONHEART's feature extraction pipeline for cancer prediction from chromatin accessibility data.

## Overview

This implementation reproduces LIONHEART's feature extraction process, which converts BAM files into features for cancer prediction. The pipeline processes DNase-seq or ATAC-seq data and computes correlation features between sample coverage and cell type chromatin masks.

## Submission package (core deliverables)

- `run_pipeline_all_chroms.py` and the modules in `src/`: end-to-end feature extraction and shared utilities (GC correction, mask loading, running stats, etc.).
- `scripts/reorder_features.py`: converts the CSV output into the `(10, 898)` matrix layout expected by the official tooling.
- `scripts/compare_feature_dataset.py`: aligns the reproduction output against the official `feature_dataset.npy` for validation.
- `artifacts/20251130/`: frozen outputs for this submission (CSV/NPY features, reordered matrices, `feature_dataset.npy`, and the prediction CSV produced with the official model).

All other helper scripts (e.g., `scripts/compare_dnase_only.py`, `scripts/list_top_diffs.py`, and the various logs/debug dumps) remain in-place for ongoing investigation but are not part of the minimal submission set. They can be archived or cleaned up later without affecting the core workflow.

## Core Components

### Main Pipeline
- **`run_pipeline_all_chroms.py`** - Main pipeline script for feature extraction across all 22 autosomes

### Source Modules (`src/`)
- `compute_coverage.py` - Coverage calculation and normalization
- `extract_insert_size.py` - Fragment length extraction and correction
- `compute_correlation.py` - Feature computation (Pearson R, p-value, etc.)
- `load_masks.py` - Loading LIONHEART chromatin masks
- `utils.py` - Utility functions (GC content, etc.)
- `blacklist.py` - Blacklist/exclude region handling
- `outlier_detection.py` - ZIPoisson outlier detection
- `insert_size_correction.py` - Insert size bias correction
- `running_stats.py` - Running statistics accumulation
- `paths.py` - Path configuration
- `optimized.py` - Numba-optimized functions

## Setup

### 1. Install Dependencies

```bash
conda create -n lionheart python=3.9
conda activate lionheart
pip install pysam numpy pandas scipy scikit-learn numba pyarrow
```

### 2. Download Data

See `DATA_SETUP.md` for instructions on:
- Downloading reference genome (hg38)
- Obtaining BAM files
- Setting up LIONHEART resource files

### 3. Configure Paths

Edit `src/paths.py` to set:
- BAM file path
- Reference genome directory
- LIONHEART resources directory

## Usage

### Run Feature Extraction

#### 1. Generate Sparse Coverage/Insert-Size Inputs (official LIONHEART)

Run the official CLI once per BAM with `--keep_intermediates` so the mosdepth-derived sparse arrays are preserved:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lionheart
lionheart extract_features \
  --bam_file /path/to/sample.bam \
  --resources_dir /path/to/lionheart_resources \
  --out_dir /path/to/lionheart_work/output_debug \
  --mosdepth_path /path/to/mosdepth \
  --ld_library_path /path/to/conda/env/lib \
  --keep_intermediates \
  --n_jobs 10
```

Key outputs used by the reproduction pipeline:
- `coverage/sparse_coverage_by_chromosome/chr*.npz`
- `coverage/sparse_insert_sizes_by_chromosome/chr*.npz`
- `dataset/feature_dataset.npy` (official reference for comparison)

#### 2. Run the Reproduction Pipeline (DNase + ATAC)

Set environment variables so the pipeline reads the official sparse coverage:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lionheart

DNASE_MASK_WORKERS=1 \
ATAC_MASK_WORKERS=1 \
COVERAGE_SPARSE_DIR=/path/to/output_debug/coverage/sparse_coverage_by_chromosome \
INSERT_SIZE_SPARSE_DIR=/path/to/output_debug/coverage/sparse_insert_sizes_by_chromosome \
python run_pipeline_all_chroms.py
```

This will:
1. Process all 22 autosomes (respecting `RUN_CHROMS` if set)
2. Apply GC, insert-size, megabin corrections matching LIONHEART
3. Stream DNase and ATAC masks (consensus removal and NaN propagation aligned with reference)
4. Write `features_all_22chroms_10bp.csv`

Feature names now include the mask type prefix (`DNase::` or `ATAC::`) to keep DNase/ATAC channels distinct, e.g. `pearson_r_DNase::8988t__cancer_cl`.

If you only want a subset (e.g., quick validation on `chr21` and one cell type):

```bash
RUN_CHROMS=chr21 \
CELL_TYPE_FILTER=8988t__cancer_cl \
python run_pipeline_all_chroms.py
```

#### 3. Compare Against Official Features

Use the helper script to align DNase + ATAC order and quantify differences:

```bash
python compare_feature_dataset.py \
  --official lionheart_reference_features/feature_dataset.npy \
  --reproduction_csv features_all_22chroms_10bp.csv \
  --cell_types \
      data/lionheart_resources/DNase.idx_to_cell_type.csv \
      data/lionheart_resources/ATAC.idx_to_cell_type.csv \
  --mask_labels DNase ATAC \
  --output_npy features_all_22chroms_10bp.npy
```

The script prints max/mean differences and optionally saves the stacked reproduction matrix for archiving.

### Runtime Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `COVERAGE_SPARSE_DIR` | Directory with `chr*.npz` sparse coverage (from mosdepth) | `resources/sparse_coverage_by_chromosome` if present |
| `INSERT_SIZE_SPARSE_DIR` | Directory with sparse insert-size sums | `resources/insert_sizes_sparse` if present |
| `RUN_CHROMS` | Comma-separated chromosome list to process (e.g., `chr21,chr22`) | All autosomes |
| `CELL_TYPE_FILTER` / `DNASE_CELL_TYPE_FILTER` | Restrict DNase processing to specific cell types (comma-separated) | All DNase indices |
| `ATAC_CELL_TYPE_FILTER` | Restrict ATAC processing to specific cell types | All ATAC indices |
| `MASK_TYPES` | Subset mask types (comma-separated subset of `DNase,ATAC`) | Both |
| `DNASE_MASK_WORKERS` | Worker count for DNase masks (set `1` to force sequential streaming) | 4 |
| `ATAC_MASK_WORKERS` | Worker count for ATAC masks (set `1` to stream, recommended on low-RAM systems) | 1 |
| `DEBUG_DUMP_CHROM`, `DEBUG_DUMP_CELL_TYPE`, `DEBUG_DUMP_DIR` | Enable per-chromosome debug dumps (mirrors LIONHEART’s debugging hooks) | disabled |

Additional tips:
- Sequential mask processing (`*_MASK_WORKERS=1`) minimizes memory when loading 354 DNase or 544 ATAC masks.
- Keep the mosdepth sparse directories on a fast disk; both pipelines read the same `.npz` files for perfect parity.
- Use `MASK_TYPES=ATAC` together with `ATAC_CELL_TYPE_FILTER=cell_a,cell_b` (and optionally `RUN_CHROMS`) to debug a small subset of ATAC masks without touching DNase.

## Output Format

The pipeline generates a CSV file with features for each cell type:
- `pearson_r_{cell_type}` - Pearson correlation coefficient
- `p_value_{cell_type}` - Statistical significance
- `fraction_within_{cell_type}` - Normalized dot product
- `cosine_similarity_{cell_type}` - Cosine similarity
- Additional statistics: `x_sum`, `y_sum`, `x_squared_sum`, `y_squared_sum`, `xy_sum`, `n`

## Pipeline Steps

1. **Coverage Calculation** - Compute coverage at 10bp resolution
2. **Fragment Length Extraction** - Extract per-bin fragment lengths
3. **ZIPoisson Clipping** - Remove extreme outliers
4. **GC Correction** - Correct GC bias
5. **Insert Size Correction** - Correct fragment length bias
6. **Megabin Normalization** - Normalize for copy number alterations
7. **Feature Computation** - Calculate correlation features with cell type masks

## Resources

- `DATA_SETUP.md` - Data download and setup instructions
- `RESOURCES_SETUP.md` - LIONHEART resource files setup

## Archive

Development, debugging, and comparison scripts have been archived in the `archive/` directory:
- `archive/comparison_scripts/` - Comparison with LIONHEART results
- `archive/utility_scripts/` - Development utilities
- `archive/debugging/` - Debugging scripts
- `archive/testing/` - Test scripts
- `archive/investigation/` - Investigation scripts
- `archive/logs/` - Log files
- `archive/outputs/` - Historical output files

## License

This is a research reproduction project. Please refer to LIONHEART's original license for the base methodology.
