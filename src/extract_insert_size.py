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


def extract_fragment_lengths_per_bin(bam_path, chrom, bin_size=100, min_len=100, max_len=220):
    """
    Extract fragment lengths per bin (not just global statistics).
    
    Returns:
        mean_frag_len: array of mean fragment length per bin
        median_frag_len: array of median fragment length per bin
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    
    try:
        chrom_len = bam.get_reference_length(chrom)
        n_bins = chrom_len // bin_size + 1
        
        # Store fragment lengths for each bin
        frag_lengths_per_bin = [[] for _ in range(n_bins)]
        
        for read in bam.fetch(chrom):
            if not read.is_proper_pair or not read.is_read1:
                continue
            
            frag_len = abs(read.template_length)
            if not (min_len <= frag_len <= max_len):
                continue
            
            start = read.reference_start
            if start < 0:
                continue
            
            bin_id = start // bin_size
            if 0 <= bin_id < n_bins:
                frag_lengths_per_bin[bin_id].append(frag_len)
        
        # Calculate mean and median for each bin
        mean_frag_len = np.zeros(n_bins, dtype=np.float64)
        median_frag_len = np.zeros(n_bins, dtype=np.float64)
        
        for bin_id in range(n_bins):
            lengths = frag_lengths_per_bin[bin_id]
            if len(lengths) > 0:
                mean_frag_len[bin_id] = np.mean(lengths)
                median_frag_len[bin_id] = np.median(lengths)
            else:
                mean_frag_len[bin_id] = np.nan
                median_frag_len[bin_id] = np.nan
        
        return mean_frag_len, median_frag_len
    
    finally:
        bam.close()

