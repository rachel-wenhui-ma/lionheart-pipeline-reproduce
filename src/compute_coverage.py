# src/compute_coverage.py
import pysam
import numpy as np

def compute_coverage(bam_path, chrom="chr21", bin_size=100):
    """
    Compute simple coverage per bin. 
    (Real LIONHEART uses mosdepth; this is simplified)
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    chrom_len = bam.get_reference_length(chrom)

    n_bins = chrom_len // bin_size + 1
    coverage = np.zeros(n_bins, dtype=np.int32)

    for read in bam.fetch(chrom):
        start = read.reference_start
        if start < 0:
            continue
        
        bin_id = start // bin_size
        if 0 <= bin_id < n_bins:
            coverage[bin_id] += 1

    bam.close()
    return coverage


def normalize_coverage(coverage):
    """Optional simple normalization."""
    return (coverage - np.mean(coverage)) / np.std(coverage)

