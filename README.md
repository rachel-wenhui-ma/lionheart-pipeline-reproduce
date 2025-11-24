# LIONHEART Feature Extraction Pipeline Reproduction

This repository contains a reproduction of LIONHEART's feature extraction pipeline for cancer prediction from chromatin accessibility data.

## Overview

This implementation reproduces LIONHEART's feature extraction process, which converts BAM files into features for cancer prediction. The pipeline processes DNase-seq or ATAC-seq data and computes correlation features between sample coverage and cell type chromatin masks.

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

```bash
python run_pipeline_all_chroms.py
```

This will:
1. Process all 22 autosomes
2. Apply corrections (GC, insert size, megabin normalization)
3. Compute correlation features with cell type masks
4. Output features to CSV file

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
