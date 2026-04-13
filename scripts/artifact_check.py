import pandas as pd
import numpy as np
from astropy.io import fits
from tqdm import tqdm

CLUSTER_FILE = 'results/anomaly_clusters_stable.parquet'
FLUX_FILE    = 'data/processed/flux_matrix.parquet'
OUT_FILE     = 'results/anomaly_clusters_checked.parquet'

SPATIAL_THRESHOLD  = 0.8   # >80% on one CCD = likely instrumental
TEMPORAL_THRESHOLD = 0.3   # >30% of anomaly signal overlaps global spikes = reject


# ── Check 1: Spatial CCD distribution ────────────────────────────────────────

def get_ccd_info(filepath):
    try:
        with fits.open(filepath) as hdul:
            camera = hdul[0].header.get('CAMERA', -1)
            ccd    = hdul[0].header.get('CCD', -1)
            sector = hdul[0].header.get('SECTOR', -1)
        return camera, ccd, sector
    except Exception:
        return -1, -1, -1

def spatial_artifact_check(df):
    print("\nCheck 1: Spatial CCD distribution...")
    cameras, ccds, sectors = [], [], []
    for fp in tqdm(df['filepath'], desc="Reading FITS headers"):
        c, d, s = get_ccd_info(fp)
        cameras.append(c)
        ccds.append(d)
        sectors.append(s)

    df['camera'] = cameras
    df['ccd']    = ccds
    df['sector'] = sectors

    spatial_flags = {}
    for cid in sorted(df['cluster'].unique()):
        if cid < 0:
            continue
        sub   = df[df['cluster'] == cid]
        total = len(sub)
        cam_ccd   = sub.groupby(['camera', 'ccd']).size()
        top_frac  = cam_ccd.max() / total
        is_artifact = top_frac > SPATIAL_THRESHOLD
        spatial_flags[cid] = is_artifact
        flag = "⚠️  SPATIAL ARTIFACT" if is_artifact else "✅ Spatially distributed"
        print(f"  Cluster {cid} (n={total}): {top_frac:.0%} on one CCD — {flag}")

    df['spatial_artifact'] = df['cluster'].map(
        lambda c: spatial_flags.get(c, False) if c >= 0 else True
    )
    return df


# ── Check 2: Temporal spike alignment ────────────────────────────────────────

def temporal_artifact_check(df, flux_df):
    """
    Build a global spike map: for each time index, count how many stars
    have an unusually high delta flux value. If many stars spike together,
    that time index is instrumental. Stars whose anomaly overlaps heavily
    with global spike times get flagged.
    """
    print("\nCheck 2: Temporal spike alignment...")

    # Use delta flux (the removed signal) for this check
    if 'flux_delta' not in flux_df.columns:
        print("  flux_delta not available — skipping temporal check")
        df['temporal_artifact_score'] = 0.0
        return df

    # Merge anomaly stars with their flux
    anomaly_ids    = set(df['tic_id'].astype(str))
    flux_sub       = flux_df[flux_df['tic_id'].astype(str).isin(anomaly_ids)]
    delta_matrix   = np.stack(flux_sub['flux_delta'].values)  # (N_stars, 1024)
    n_stars, n_pts = delta_matrix.shape

    # Define a spike as > 3 std above the mean for that star
    means = delta_matrix.mean(axis=1, keepdims=True)
    stds  = delta_matrix.std(axis=1, keepdims=True)
    stds  = np.where(stds == 0, 1e-9, stds)
    spike_matrix = ((delta_matrix - means) / stds) > 3.0  # (N_stars, 1024) bool

    # Global spike map: fraction of stars spiking at each time index
    global_spike_fraction = spike_matrix.mean(axis=0)  # (1024,)

    # Flag time indices where >20% of stars spike simultaneously
    global_spike_mask = global_spike_fraction > 0.20
    n_flagged_times   = global_spike_mask.sum()
    print(f"  Global spike timestamps: {n_flagged_times}/{n_pts} "
          f"({n_flagged_times/n_pts:.1%})")

    # For each star: what fraction of its own spikes overlap with global spikes?
    temporal_scores = {}
    tic_ids_sub     = flux_sub['tic_id'].astype(str).values
    for i, tic_id in enumerate(tic_ids_sub):
        own_spikes     = spike_matrix[i]
        own_spike_count = own_spikes.sum()
        if own_spike_count == 0:
            temporal_scores[tic_id] = 0.0
        else:
            overlap = (own_spikes & global_spike_mask).sum()
            temporal_scores[tic_id] = overlap / own_spike_count

    df['temporal_artifact_score'] = df['tic_id'].astype(str).map(
        lambda t: temporal_scores.get(t, 0.0)
    )
    df['temporal_artifact'] = df['temporal_artifact_score'] > TEMPORAL_THRESHOLD

    # Summary per cluster
    for cid in sorted(df['cluster'].unique()):
        if cid < 0:
            continue
        sub     = df[df['cluster'] == cid]
        frac    = sub['temporal_artifact'].mean()
        flag    = "⚠️  TEMPORAL ARTIFACT" if frac > 0.5 else "✅ Temporally clean"
        print(f"  Cluster {cid}: {frac:.0%} of stars have temporal overlap — {flag}")

    return df


def main():
    df      = pd.read_parquet(CLUSTER_FILE)
    flux_df = pd.read_parquet('data/processed/flux_matrix.parquet',
                               columns=['tic_id', 'flux_delta'])

    df = spatial_artifact_check(df)
    df = temporal_artifact_check(df, flux_df)

    # Combined artifact flag
    df['artifact_flagged'] = df['spatial_artifact'] | df['temporal_artifact']

    clean_clusters = df[~df['artifact_flagged'] & (df['cluster'] >= 0)]['cluster'].nunique()
    print(f"\n{clean_clusters} clusters passed both artifact checks")

    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")

if __name__ == '__main__':
    main()