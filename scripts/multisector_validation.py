"""
Phase L — Multi-Sector Repeatability Validation
For each anomaly star that passed artifact checks, fetches light curves from
sectors outside the training set (1–5) and checks whether the anomaly repeats.
A star is considered anomalous in a secondary sector if its delta std exceeds
the P75 of the original anomaly population's delta std (data-derived threshold).
Output: results/anomaly_clusters_validated.parquet
"""

import numpy as np
import pandas as pd
import lightkurve as lk
from scipy.interpolate import interp1d
from tqdm import tqdm

CLUSTER_FILE = 'results/anomaly_clusters_checked.parquet'  # 88 stars (Teff<7000, RUWE<8)
FLUX_FILE    = 'data/processed/flux_matrix.parquet'
OUT_FILE     = 'results/anomaly_clusters_validated.parquet'

N_POINTS          = 1024
REPEATABILITY_MIN = 0.30   # Star needs anomaly in ≥30% of observed sectors to count
MAX_EXTRA_SECTORS = 5      # Check up to this many additional sectors per star
ORIG_SECTORS      = set(range(1, 6))  # Sectors used in training


def normalize_and_resample(time, flux, n_points=N_POINTS):
    flux = flux.astype(np.float64)
    good = np.isfinite(flux) & np.isfinite(time)
    time, flux = time[good], flux[good]
    if len(flux) < 100:
        return None
    mean, std = np.mean(flux), np.std(flux)
    if std == 0 or not np.isfinite(std):
        return None
    flux_norm = (flux - mean) / std
    t_min, t_max = time[0], time[-1]
    if t_max <= t_min:
        return None
    t_uniform = np.linspace(t_min, t_max, n_points)
    try:
        f = interp1d(time, flux_norm, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        return f(t_uniform)
    except Exception:
        return None


def compute_delta_strength(tic_id, sector):
    """
    Download the SPOC light curve for tic_id/sector and return the std of
    (normalised SAP − normalised PDCSAP). Returns None if unavailable.
    """
    try:
        search = lk.search_lightcurve(
            f'TIC {tic_id}', mission='TESS', author='SPOC',
            sector=sector, exptime=120
        )
        if len(search) == 0:
            return None

        lc      = search[0].download(flux_column='sap_flux')
        lc_full = search[0].download()
        if lc is None or lc_full is None:
            return None

        sap_r    = normalize_and_resample(lc.time.value, lc.flux.value)
        pdcsap_r = normalize_and_resample(lc_full.time.value, lc_full.flux.value)
        if sap_r is None or pdcsap_r is None:
            return None

        return float(np.std(sap_r - pdcsap_r))

    except Exception:
        return None


def get_all_sectors_for_tic(tic_id):
    """Return sorted list of TESS SPOC 2-min sectors for this TIC ID."""
    try:
        results = lk.search_lightcurve(
            f'TIC {tic_id}', mission='TESS', author='SPOC', exptime=120
        )
        if len(results) == 0:
            return []
        sectors = []
        for mission_str in results.table['mission']:
            try:
                s = int(str(mission_str).split()[-1])
                sectors.append(s)
            except Exception:
                pass
        return sorted(set(sectors))   # sorted → deterministic sector selection
    except Exception:
        return []


def derive_delta_threshold(flux_file, anomaly_tic_ids):
    """
    Compute P90 of delta-std across the original anomaly population.
    This is the data-derived threshold a secondary sector must exceed
    for a star to be counted as anomalous there.
    """
    flux_df = pd.read_parquet(flux_file, columns=['tic_id', 'flux_sap', 'flux_pdcsap'])
    sub     = flux_df[flux_df['tic_id'].astype(str).isin(anomaly_tic_ids)]
    delta_stds = []
    for _, row in sub.iterrows():
        sap    = np.array(row['flux_sap'],    dtype=np.float32)
        pdcsap = np.array(row['flux_pdcsap'], dtype=np.float32)
        delta_stds.append(float(np.std(sap - pdcsap)))
    threshold = float(np.percentile(delta_stds, 75))
    print(f"  Delta-std P75 across {len(delta_stds)} anomaly stars: {threshold:.4f}")
    return threshold


def main():
    df = pd.read_parquet(CLUSTER_FILE)

    candidates = df[~df['artifact_flagged'] & (df['cluster'] >= 0)].copy()
    print(f"Validating {len(candidates)} stars across "
          f"{candidates['cluster'].nunique()} clusters")

    # Derive the anomaly threshold from the actual data distribution
    print("Deriving repeatability threshold from original anomaly population...")
    anomaly_ids       = set(candidates['tic_id'].astype(str))
    delta_threshold   = derive_delta_threshold(FLUX_FILE, anomaly_ids)
    print(f"  A secondary sector is 'anomalous' if delta_std > {delta_threshold:.4f} (P75 threshold)")

    repeatability_results = []

    for _, row in tqdm(candidates.iterrows(), total=len(candidates),
                       desc="Multi-sector validation"):
        tic_id = str(row['tic_id'])

        all_sectors   = get_all_sectors_for_tic(tic_id)
        other_sectors = [s for s in all_sectors if s not in ORIG_SECTORS]

        if not other_sectors:
            repeatability_results.append({
                'tic_id': tic_id,
                'sectors_checked': 0,
                'sectors_anomalous': 0,
                'repeatability_score': 0.0
            })
            continue

        n_anomalous = 0
        n_checked   = 0

        for sec in other_sectors[:MAX_EXTRA_SECTORS]:
            delta_strength = compute_delta_strength(tic_id, sec)
            if delta_strength is None:
                continue
            n_checked += 1
            if delta_strength > delta_threshold:
                n_anomalous += 1

        repeatability_score = n_anomalous / n_checked if n_checked > 0 else 0.0
        repeatability_results.append({
            'tic_id': tic_id,
            'sectors_checked': n_checked,
            'sectors_anomalous': n_anomalous,
            'repeatability_score': repeatability_score
        })

    rep_df = pd.DataFrame(repeatability_results)
    df     = df.merge(rep_df, on='tic_id', how='left')

    df['repeatability_score'] = df['repeatability_score'].fillna(0.0)
    df['sectors_checked']     = df['sectors_checked'].fillna(0).astype(int)
    df['sectors_anomalous']   = df['sectors_anomalous'].fillna(0).astype(int)

    print("\nCluster repeatability summary:")
    for cid in sorted(df['cluster'].unique()):
        if cid < 0:
            continue
        sub     = df[(df['cluster'] == cid) & (~df['artifact_flagged'])]
        if len(sub) == 0:
            continue
        checked = sub[sub['sectors_checked'] > 0]
        if len(checked) == 0:
            print(f"  Cluster {cid}: no multi-sector data available")
            continue
        pct_repeatable = (checked['repeatability_score'] >= REPEATABILITY_MIN).mean()
        flag = "🔴 DISCOVERY CANDIDATE" if pct_repeatable >= 0.3 else \
               "🟡 Partial repeatability" if pct_repeatable > 0 else \
               "⚪ Not repeatable"
        print(f"  Cluster {cid}: {pct_repeatable:.0%} of checked stars repeatable — {flag}")

    df.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == '__main__':
    main()
