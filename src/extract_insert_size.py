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


def extract_fragment_lengths_per_bin(bam_path, chrom, bin_size=100, min_len=100, max_len=220, mapq_threshold=20, use_overlap=True):
    """
    Extract fragment lengths per bin using overlap-based method (matching mosdepth --insert-size-mode).
    
    mosdepth --insert-size-mode outputs sum of insert sizes per bin (weighted by overlap),
    then LIONHEART divides by coverage to get mean. This function implements the same logic.
    
    Parameters:
        bam_path: Path to BAM file
        chrom: Chromosome name
        bin_size: Bin size in bp (default 100)
        min_len: Minimum fragment length (default 100)
        max_len: Maximum fragment length (default 220)
        mapq_threshold: Minimum MAPQ score (default 20, matching LIONHEART)
        use_overlap: If True, use overlap-based calculation (mosdepth-like). If False, use read start only.
    
    Returns:
        mean_frag_len: array of mean fragment length per bin
        median_frag_len: array of median fragment length per bin (not used by LIONHEART, but kept for compatibility)
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    
    try:
        chrom_len = bam.get_reference_length(chrom)
        n_bins = chrom_len // bin_size + 1
        
        if use_overlap:
            # Overlap-based method (matching mosdepth --insert-size-mode)
            # mosdepth sums insert sizes weighted by overlap fraction
            frag_sums = np.zeros(n_bins, dtype=np.float64)
            overlap_sums = np.zeros(n_bins, dtype=np.float64)  # For calculating mean
            
            for read in bam.fetch(chrom):
                # Apply MAPQ filter (matching LIONHEART's --mapq 20)
                if read.mapping_quality < mapq_threshold:
                    continue
                
                if not read.is_proper_pair or not read.is_read1:
                    continue
                
                frag_len = abs(read.template_length)
                if not (min_len <= frag_len <= max_len):
                    continue
                
                read_start = read.reference_start
                read_end = read.reference_end
                
                if read_start < 0 or read_end is None:
                    continue
                
                # Find all bins this read overlaps
                start_bin = read_start // bin_size
                end_bin = read_end // bin_size if read_end else start_bin
                
                # Calculate overlap fraction for each overlapping bin
                for bin_id in range(start_bin, min(end_bin + 1, n_bins)):
                    bin_start = bin_id * bin_size
                    bin_end = (bin_id + 1) * bin_size
                    
                    # Calculate overlap
                    overlap_start = max(read_start, bin_start)
                    overlap_end = min(read_end, bin_end)
                    overlap_length = max(0, overlap_end - overlap_start)
                    overlap_fraction = overlap_length / bin_size
                    
                    # Add weighted insert size (frag_len * overlap_fraction)
                    # This matches mosdepth: sum of insert sizes weighted by overlap
                    frag_sums[bin_id] += frag_len * overlap_fraction
                    overlap_sums[bin_id] += overlap_fraction
            
            # Calculate mean: sum / overlap_sum (equivalent to dividing by coverage)
            mean_frag_len = np.zeros(n_bins, dtype=np.float64)
            for bin_id in range(n_bins):
                if overlap_sums[bin_id] > 0:
                    mean_frag_len[bin_id] = frag_sums[bin_id] / overlap_sums[bin_id]
                else:
                    mean_frag_len[bin_id] = np.nan
            
            # Median not used by LIONHEART, but return for compatibility
            median_frag_len = np.full(n_bins, np.nan, dtype=np.float64)
            
        else:
            # Original method: use read start position only
            frag_lengths_per_bin = [[] for _ in range(n_bins)]
            
            for read in bam.fetch(chrom):
                if read.mapping_quality < mapq_threshold:
                    continue
                
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

