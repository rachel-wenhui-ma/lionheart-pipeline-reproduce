# LIONHEART Feature Extraction Pipeline - Simplified Reproduction

A simplified implementation of the LIONHEART feature extraction pipeline for cfDNA cancer detection.

## Overview

This is the initial skeleton version of the pipeline, demonstrating the basic structure:
1. Fragment length extraction
2. Coverage computation
3. Correlation with cell-type masks
4. Feature assembly

## Structure

```
reproduce/
├── run_pipeline.py          # Main pipeline script
├── src/
│   ├── extract_insert_size.py
│   ├── compute_coverage.py
│   ├── compute_correlation.py
│   ├── assemble_features.py
│   └── utils.py
├── data/
│   └── masks/               # Example masks (Tcell, Monocyte, Liver)
└── README.md
```

## Requirements

- Python 3.8+
- pysam
- numpy
- scipy

## Usage

Update the BAM file path in `run_pipeline.py`, then:

```bash
python run_pipeline.py
```

## Features

- Fragment length extraction
- Coverage computation
- Simple z-score normalization
- Pearson correlation with cell-type masks

## License

[Your license here]
