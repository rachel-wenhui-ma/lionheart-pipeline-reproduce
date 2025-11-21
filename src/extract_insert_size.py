# src/extract_insert_size.py
import pysam
import numpy as np

def extract_fragment_lengths(bam_path, chrom=None, min_len=100, max_len=220):
    """
    Reads a BAM file and extracts fragment lengths within [min_len, max_len].
    If chrom is specified, only extracts from that chromosome.
    """
    bam = pysam.AlignmentFile(bam_path, "rb")

    lengths = []
    try:
        if chrom:
            reads = bam.fetch(chrom)
        else:
            reads = bam.fetch()
        
        for read in reads:
            if not read.is_proper_pair:
                continue
            if read.is_read1:
                l = abs(read.template_length)
                if min_len <= l <= max_len:
                    lengths.append(l)
    finally:
        bam.close()
    
    return np.array(lengths)


def compute_fragment_features(lengths):
    """Compute simple summary statistics."""
    return {
        "frag_mean": float(np.mean(lengths)),
        "frag_median": float(np.median(lengths)),
        "frag_std": float(np.std(lengths)),
    }

