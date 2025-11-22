# src/utils.py
import numpy as np
import pysam

def summarize_coverage(coverage):
    return {
        "cov_mean": float(np.mean(coverage)),
        "cov_std": float(np.std(coverage)),
        "cov_median": float(np.median(coverage)),
    }


def calculate_gc_content(sequence):
    """
    Calculate GC content (percentage of G and C bases) for a DNA sequence.
    """
    seq_upper = sequence.upper()
    if len(seq_upper) == 0:
        return 0.0
    gc_count = seq_upper.count('G') + seq_upper.count('C')
    return gc_count / len(seq_upper)


def calculate_gc_content_per_bin(reference_fasta, chrom, bin_size=100):
    """
    Calculate GC content for each bin from reference genome.
    
    Returns:
        gc_content: array of GC content (0-1) for each bin
    """
    ref = pysam.FastaFile(reference_fasta)
    try:
        chrom_len = ref.get_reference_length(chrom)
        n_bins = chrom_len // bin_size + 1
        gc_content = np.zeros(n_bins, dtype=np.float64)
        
        for bin_id in range(n_bins):
            start = bin_id * bin_size
            end = min(start + bin_size, chrom_len)
            if start >= chrom_len:
                break
            
            seq = ref.fetch(chrom, start, end)
            gc_content[bin_id] = calculate_gc_content(seq)
        
        return gc_content
    finally:
        ref.close()

