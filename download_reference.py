"""
Download hg38 reference genome (or single chromosome for testing).
"""
import urllib.request
import os
from pathlib import Path

def download_chr21():
    """Download only chr21 for testing (much smaller, ~50MB)."""
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz"
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "chr21.fa.gz"
    output_fasta = output_dir / "chr21.fa"
    
    if output_fasta.exists():
        print(f"[OK] {output_fasta} already exists")
        return str(output_fasta)
    
    print(f"Downloading chr21 from {url}...")
    print("This may take a few minutes (~50MB)...")
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"[OK] Downloaded to {output_file}")
        
        # Decompress
        import gzip
        print("Decompressing...")
        with gzip.open(output_file, 'rb') as f_in:
            with open(output_fasta, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Remove compressed file
        output_file.unlink()
        print(f"[OK] Decompressed to {output_fasta}")
        return str(output_fasta)
    
    except Exception as e:
        print(f"Error downloading: {e}")
        raise


def download_full_hg38():
    """Download full hg38 reference genome (~3GB compressed)."""
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "hg38.fa.gz"
    output_fasta = output_dir / "hg38.fa"
    
    if output_fasta.exists():
        print(f"[OK] {output_fasta} already exists")
        return str(output_fasta)
    
    print(f"Downloading full hg38 from {url}...")
    print("WARNING: This is a large file (~3GB compressed, ~9GB uncompressed)")
    print("This may take a very long time...")
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"[OK] Downloaded to {output_file}")
        
        # Decompress
        import gzip
        print("Decompressing (this will take a while)...")
        with gzip.open(output_file, 'rb') as f_in:
            with open(output_fasta, 'wb') as f_out:
                while True:
                    chunk = f_in.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f_out.write(chunk)
        
        # Remove compressed file
        output_file.unlink()
        print(f"[OK] Decompressed to {output_fasta}")
        return str(output_fasta)
    
    except Exception as e:
        print(f"Error downloading: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # Download full genome
        download_full_hg38()
    else:
        # Download only chr21 for testing
        print("Downloading chr21 only (for testing)...")
        print("Use 'python download_reference.py full' for full hg38")
        print()
        ref_path = download_chr21()
        print(f"\n[OK] Reference genome ready at: {ref_path}")
        print("\nNote: Update run_pipeline.py to use 'data/chr21.fa' for chr21")

