# LIONHEART Pipeline Reproduction

This repository contains a reproduction of LIONHEART's pipeline for cancer prediction from chromatin accessibility data.

## Overview

This implementation reproduces LIONHEART's complete pipeline, consisting of two main components:

1. **Feature Extraction**: Converts BAM files into features for cancer prediction. The pipeline processes DNase-seq or ATAC-seq data and computes correlation features between sample coverage and cell type chromatin masks across 898 cell types.

2. **Model Training and Prediction**: Trains a LASSO logistic regression model using nested leave-one-dataset-out cross-validation, and applies the trained model to make cancer predictions from extracted features.

## Core Components

### Feature Extraction

#### Main Pipeline
- **`run_pipeline_all_chroms.py`** - Main pipeline script for feature extraction across all 22 autosomes

#### Source Modules (`src/`)
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

### Model Training and Prediction

#### Model Pipeline
- **`scripts/reproduce_model.py`** - Model training and prediction script that:
  - Loads features from multiple datasets
  - Performs nested leave-one-dataset-out cross-validation for hyperparameter tuning
  - Trains final LASSO logistic regression model with dataset-balanced sample weights
  - Applies PCA with variance-based component selection
  - Makes predictions on validation/test datasets

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

#### 4. Train Model and Make Predictions

Train the model using nested leave-one-dataset-out cross-validation:

```bash
python scripts/reproduce_model.py
```

This script:
- Loads features from 9 training datasets
- Performs nested cross-validation to tune hyperparameters (PCA variance and L1 regularization C)
- Trains the final model on all training data
- Evaluates on the validation dataset (Zhu Validation)
- Saves the trained model to `model/model.joblib`

The model uses:
- StandardScaler for feature normalization
- PCA with variance-based component selection (default: 98.7%-99.7% variance)
- LASSO logistic regression with dataset-balanced sample weights
- Max-J threshold selection for optimal sensitivity/specificity balance

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

## License

This is a research reproduction project. Please refer to LIONHEART's original license for the base methodology.
