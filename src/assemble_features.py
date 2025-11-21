# src/assemble_features.py
import csv

def write_feature_vector(output_path, frag_stats, cov_stats, corr_stats):
    """
    Combine all features into one flat dictionary and write to CSV.
    """
    all_feat = {}
    all_feat.update(frag_stats)
    all_feat.update(cov_stats)
    all_feat.update(corr_stats)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "value"])
        for k, v in all_feat.items():
            writer.writerow([k, v])

    print(f"[+] Wrote features to: {output_path}")

