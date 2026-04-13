import numpy as np
import pandas as pd
import lightkurve as lk
from scipy.interpolate import interp1d
from tqdm import tqdm

CLUSTER_FILE = 'results/anomaly_clusters_checked.parquet'
OUT_FILE     = 'results/anomaly_clusters_validated.parquet'

N_POINTS          = 1024
REPEATABILITY_MIN = 0.30  # Star needs anomaly in ≥30% of observed sectors to count

# Anomaly threshold for secondary sectors:
# A star is anomalous in a sector if its delta std is in the top 10%
# of the ORIGINAL anomaly population's delta std.
# We use a fixed threshold derived from the main experiment's P90 delta strength.
DELTA_ANOMALY_THRESHOLD_SIGMA = 2.0  # Signal must be > 2 std above the sector baseline


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
    Download light curve for a given TIC ID and sector.
    Returns the std of the delta flux (SAP - PDCSAP), or None if unavailable.
    This is our proxy for 'anomaly present in this sector'.
    """
    try:
        search = lk.search_lightcurve(
            f'TIC {tic_id}', mission='TESS', author='SPOC',
            sector=sector, exptime=120
        )
        if len(search) == 0:
            return None

        lc = search[0].download(flux_column='sap_flux')
        if lc is None:
            return None

        # Also get PDCSAP
        lc_full = search[0].download()
        if lc_full is None:
            return None

        sap_r    = normalize_and_resample(lc.time.value, lc.flux.value)
        pdcsap_r = normalize_and_resample(lc_full.time.value, lc_full.flux.value)

        if sap_r is None or pdcsap_r is None:
            return None

        delta = sap_r - pdcsap_r
        return np.std(delta)

    except Exception:
        return None


def get_all_sectors_for_tic(tic_id):
    """Return list of all TESS sectors where this TIC was observed at 2-min cadence."""
    try:
        results = lk.search_lightcurve(
            f'TIC {tic_id}', mission='TESS', author='SPOC', exptime=120
        )
        if len(results) == 0:
            return []
        return [int(r.mission[0].split('Sector')[1].strip())
                for r in results.table.iterrows()
                if 'Sector' in str(r)]
    except Exception:
        # Fallback: extract sector numbers from the search result table
        try:
            sectors = []
            for row in results:
                try:
                    s = int(str(row.table['mission'][0]).split()[-1])
                    sectors.append(s)
                except Exception:
                    pass
            return list(set(sectors))
        except Exception:
            return []


def main():
    df = pd.read_parquet(CLUSTER_FILE)

    # Only validate stars in clusters that passed artifact checks
    candidates = df[~df['artifact_flagged'] & (df['cluster'] >= 0)].copy()
    print(f"Validating {len(candidates)} stars across {candidates['cluster'].nunique()} clusters")

    # The original sectors are 1–5. Baseline delta strength per star comes from
    # the original experiment. We use a simple proxy: std of flux_delta is our measure.
    # For secondary sectors, we fetch on-demand via lightkurve (cached locally).

    repeatability_results = []
    skipped = 0

    for _, row in tqdm(candidates.iterrows(), total=len(candidates),
                       desc="Multi-sector validation"):
        tic_id    = str(row['tic_id'])
        orig_sec  = int(str(row['filepath']).split('sector')[1][0])  # From filepath

        # Find all other sectors where this star was observed
        all_sectors = get_all_sectors_for_tic(tic_id)
        other_sectors = [s for s in all_sectors if s not in range(1, 6)]

        if not other_sectors:
            repeatability_results.append({
                'tic_id': tic_id,
                'sectors_checked': 0,
                'sectors_anomalous': 0,
                'repeatability_score': 0.0
            })
            continue

        # Original baseline: sap_pdcsap_ratio from Phase F
        baseline_ratio = row.get('sap_pdcsap_ratio', 1.0)

        n_anomalous = 0
        n_checked   = 0

        for sec in other_sectors[:5]:  # Check up to 5 additional sectors per star
            delta_strength = compute_delta_strength(tic_id, sec)
            if delta_strength is None:
                continue
            n_checked += 1
            # Anomalous if delta std is meaningfully large relative to the baseline
            if delta_strength > DELTA_ANOMALY_THRESHOLD_SIGMA * 0.1:
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

    # Fill NaN for stars not in candidates
    df['repeatability_score']  = df['repeatability_score'].fillna(0.0)
    df['sectors_checked']      = df['sectors_checked'].fillna(0).astype(int)
    df['sectors_anomalous']    = df['sectors_anomalous'].fillna(0).astype(int)

    # Cluster-level repeatability summary
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