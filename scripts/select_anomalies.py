"""
Phase F — Select Anomalies with SAP/PDCSAP Ratio Filter
Selects top 1% of stars by combined anomaly score, then filters to those
where SAP signal is substantially stronger than PDCSAP (i.e. signal was suppressed).
Output: results/anomalies.parquet
"""

import numpy as np
import pandas as pd

ERRORS_FILE  = 'results/reconstruction_errors.parquet'
LATENT_FILE  = 'results/latent_sap.parquet'
FLUX_FILE    = 'data/processed/flux_matrix.parquet'
OUT_FILE     = 'results/anomalies.parquet'

ANOMALY_PERCENTILE = 99   # Top 1% by combined score
SAP_PDCSAP_RATIO_THRESHOLD = 1.5  # SAP signal must be at least 1.5x stronger than PDCSAP

def compute_signal_strength(flux_list):
    """Signal strength = std of the flux sequence."""
    return np.std(flux_list)

def main():
    errors  = pd.read_parquet(ERRORS_FILE)
    latents = pd.read_parquet(LATENT_FILE)
    flux_df = pd.read_parquet(FLUX_FILE, columns=['tic_id', 'flux_sap', 'flux_pdcsap'])

    # Step 1: top 1% by combined score
    threshold = np.percentile(errors['combined_score'], ANOMALY_PERCENTILE)
    print(f"Combined score threshold (P{ANOMALY_PERCENTILE}): {threshold:.6f}")
    anomalies = errors[errors['combined_score'] >= threshold].copy()
    print(f"  Stars above threshold: {len(anomalies)}")

    # Step 2: SAP/PDCSAP ratio filter
    flux_sub = flux_df[flux_df['tic_id'].isin(anomalies['tic_id'])].copy()
    flux_sub['sap_strength']    = flux_sub['flux_sap'].apply(compute_signal_strength)
    flux_sub['pdcsap_strength'] = flux_sub['flux_pdcsap'].apply(compute_signal_strength)

    # Avoid division by zero
    flux_sub['pdcsap_strength'] = flux_sub['pdcsap_strength'].replace(0, 1e-9)
    flux_sub['sap_pdcsap_ratio'] = flux_sub['sap_strength'] / flux_sub['pdcsap_strength']

    before = len(anomalies)
    passing_ids = flux_sub[flux_sub['sap_pdcsap_ratio'] >= SAP_PDCSAP_RATIO_THRESHOLD]['tic_id']
    anomalies = anomalies[anomalies['tic_id'].isin(passing_ids)].copy()
    print(f"  After SAP/PDCSAP ratio filter (≥{SAP_PDCSAP_RATIO_THRESHOLD}x): "
          f"{len(anomalies)} stars (removed {before - len(anomalies)})")

    # Attach ratio values
    anomalies = anomalies.merge(
        flux_sub[['tic_id', 'sap_strength', 'pdcsap_strength', 'sap_pdcsap_ratio']],
        on='tic_id', how='left'
    )

    # Merge with latent vectors
    anomalies = anomalies.merge(latents, on='tic_id', how='left')

    anomalies.to_parquet(OUT_FILE, index=False)
    print(f"Saved {len(anomalies)} anomalies to {OUT_FILE}")

if __name__ == '__main__':
    main()