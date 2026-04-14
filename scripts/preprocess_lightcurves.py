"""
Phase D — Preprocessing: Build the Flux Matrix
Reads all FITS light curves, extracts SAP / PDCSAP / delta flux,
normalises and resamples to N_POINTS, saves as a single parquet file.
Output: data/processed/flux_matrix.parquet
"""

import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from tqdm import tqdm

BORING_IDS_FILE = 'data/boring_tic_ids.txt'
RAW_DIR     = 'data/raw'
OUT_PARQUET = 'data/processed/flux_matrix.parquet'
N_POINTS    = 1024
QUALITY_MASK = 0  # Keep only cadences with quality flag == 0

os.makedirs('data/processed', exist_ok=True)


def read_both_fluxes(filepath):
    """
    Read SAP_FLUX and PDCSAP_FLUX from a TESS SPOC light curve FITS file.
    Returns (time, sap_flux, pdcsap_flux, tic_id) or None if unreadable.
    """
    try:
        with fits.open(filepath) as hdul:
            lc     = hdul['LIGHTCURVE'].data
            tic_id = str(hdul[0].header.get('TICID', 'unknown'))

            time   = lc['TIME'].astype(np.float64)
            sap    = lc['SAP_FLUX'].astype(np.float32)
            pdcsap = lc['PDCSAP_FLUX'].astype(np.float32)
            qual   = lc['QUALITY'].astype(np.int32)

            good = (
                (qual == QUALITY_MASK)
                & np.isfinite(sap)
                & np.isfinite(pdcsap)
                & np.isfinite(time)
            )

            time   = time[good]
            sap    = sap[good]
            pdcsap = pdcsap[good]

            if len(sap) < 200:
                return None

            return time, sap, pdcsap, tic_id
    except Exception:
        return None


def normalize(flux):
    """Zero-mean, unit-variance normalisation per star."""
    mean = np.mean(flux)
    std  = np.std(flux)
    if std == 0 or not np.isfinite(std):
        return None
    return (flux - mean) / std


def resample(time, flux, n_points=N_POINTS):
    """Resample to fixed length via linear interpolation on a uniform time grid."""
    t_min, t_max = time[0], time[-1]
    if t_max <= t_min:
        return None
    t_uniform = np.linspace(t_min, t_max, n_points)
    try:
        f   = interp1d(time, flux, kind='linear', bounds_error=False,
                       fill_value='extrapolate')
        out = f(t_uniform).astype(np.float32)
    except Exception:
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out


def main():
    fits_files = glob.glob(os.path.join(RAW_DIR, '**', '*.fits'), recursive=True)
    print(f"Found {len(fits_files)} FITS files")
    
    # Load boring IDs to exclude known variables from the flux matrix
    with open(BORING_IDS_FILE) as f:
        boring_ids = set(line.strip() for line in f if line.strip())
    print(f"Boring IDs loaded: {len(boring_ids)}")

    records = []
    skipped = 0

    for filepath in tqdm(fits_files, desc="Preprocessing"):
        result = read_both_fluxes(filepath)
        if result is None:
            skipped += 1
            continue

        time, sap, pdcsap, tic_id = result
        
        if tic_id not in boring_ids:
            skipped += 1
            continue

        sap_norm    = normalize(sap)
        pdcsap_norm = normalize(pdcsap)
        if sap_norm is None or pdcsap_norm is None:
            skipped += 1
            continue

        # Delta: what the pipeline removed, normalised independently
        delta     = sap_norm - pdcsap_norm
        delta_std = np.std(delta)
        if delta_std == 0 or not np.isfinite(delta_std):
            skipped += 1
            continue
        delta_norm = (delta - np.mean(delta)) / delta_std

        sap_r    = resample(time, sap_norm)
        pdcsap_r = resample(time, pdcsap_norm)
        delta_r  = resample(time, delta_norm)

        if sap_r is None or pdcsap_r is None or delta_r is None:
            skipped += 1
            continue

        records.append({
            'tic_id':      tic_id,
            'filepath':    filepath,
            'flux_sap':    sap_r.tolist(),
            'flux_pdcsap': pdcsap_r.tolist(),
            'flux_delta':  delta_r.tolist(),
        })

    print(f"Processed: {len(records)} stars | Skipped: {skipped}")
    df = pd.DataFrame(records)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved to {OUT_PARQUET}  ({len(df)} stars × {N_POINTS} pts × 3 channels)")


if __name__ == '__main__':
    main()
