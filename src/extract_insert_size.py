# src/extract_insert_size.py
import pysam
import numpy as np

def extract_fragment_lengths(bam_path, min_len=100, max_len=220):
    """
    Reads a BAM file and extracts fragment lengths within [min_len, max_len].
    """
    bam = pysam.AlignmentFile(bam_path, "rb")

    lengths = []
    for read in bam.fetch():
        if not read.is_proper_pair:
            continue
        if read.is_read1:
            l = abs(read.template_length)
            if min_len <= l <= max_len:
                lengths.append(l)

    bam.close()
    return np.array(lengths)


def compute_fragment_features(lengths):
    """Compute simple summary statistics."""
    return {
        "frag_mean": float(np.mean(lengths)),
        "frag_median": float(np.median(lengths)),
        "frag_std": float(np.std(lengths)),
    }

