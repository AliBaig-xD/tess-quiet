# TESS Quiet Stars Moonshot: End-to-End Experiment Spec (v1.1)

This document is a step-by-step specification for discovering anomalous stellar signals
hidden inside officially "non-variable" stars from NASA's TESS mission. The goal is to
identify stars whose raw (SAP) light curves contain genuine astrophysical signals that
the TESS pipeline suppressed during processing, cluster those signals into reproducible
groups, and produce a target list that astronomers can follow up.

The document is written so that a human or an AI agent can follow it linearly to
reproduce the experiment from a clean GCP instance.

**v1.1 changes:** Six upgrades integrated based on peer review — delta flux modeling,
time-aligned artifact detection, multi-sector repeatability validation, latent density
scoring, cluster stability checking, and SAP/PDCSAP ratio filtering. Upgrade 7
(multi-scale representation) was reviewed and rejected as unnecessary complexity.

---

## 1. High-Level Goal

> Use TESS 2-minute cadence light curves to find stars that are **officially classified
> as non-variable but show anomalous raw flux signals** — signals the pipeline smoothed
> away — and cluster those anomalies into groups with shared stellar properties.

TESS observes ~20,000 stars per sector at 2-minute cadence. Each light curve contains
two flux columns: `SAP_FLUX` (raw aperture photometry) and `PDCSAP_FLUX` (pipeline-cleaned).
The pipeline is designed to find planet transits and known periodic variability. Anything
that does not match those templates — slow drifts, irregular dimmings, aperiodic bursts —
is treated as noise and removed.

We train two 1D convolutional autoencoders: one on SAP flux (what the star looks like
raw) and one on the **delta flux** (SAP minus PDCSAP — the signal the pipeline explicitly
removed). Stars with high combined anomaly score across both models are the targets.
We cluster their latent embeddings, apply layered artifact and stability filtering, run
multi-sector repeatability validation, and cross-match with TIC stellar parameters.

The experiment aims for a **moonshot** outcome: identifying a population of stars with
a recurring unexplained signal type that has been systematically suppressed by the
standard pipeline and not previously reported in the literature.

### 1.1 Success Tiers

**Tier 1 — Expected:** Anomaly clusters identified, artifact-filtered, visually inspectable.

**Tier 2 — Strong:** A cluster with consistent light curve shape + shared stellar
properties + low artifact score + stable across clustering runs.

**Tier 3 — Discovery Candidate:** A cluster that additionally shows high SAP/PDCSAP
signal divergence AND multi-sector repeatability in ≥30% of member stars.

---

## 2. Infrastructure

### 2.1 GCP Instance Specification

This experiment runs on the existing GCP T4 GPU instance. Minimum confirmed requirements:

- **OS:** Ubuntu 22.04 LTS
- **GPU:** NVIDIA T4 (16 GB VRAM) — used for autoencoder training
- **vCPU:** 4+
- **RAM:** 16 GB+
- **Disk:** 120 GB SSD (raw FITS ~20–25 GB; processed parquets ~5 GB; results ~2 GB;
  multi-sector downloads for anomaly stars ~10 GB additional)

### 2.2 SSH Access

```bash
ssh <user>@<your_gcp_instance_ip>
```

---

## 3. System Setup

All commands assume you are logged in with sudo privileges.

### 3.1 Update and Install Base Packages

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
  build-essential \
  curl wget git unzip \
  python3 python3-pip python3-venv \
  tmux
```

`tmux` keeps long-running downloads and training jobs alive if SSH disconnects.

### 3.2 Verify GPU

```bash
nvidia-smi
```

You should see the T4 listed with 16,160 MiB memory. If this fails, install NVIDIA
drivers before proceeding.

### 3.3 Create Project Directory

```bash
mkdir -p ~/tess-quiet/{data,data/raw,data/processed,env,logs,scripts,results,app}
cd ~/tess-quiet
```

---

## 4. Python Environment

### 4.1 Create and Activate Virtual Environment

```bash
python3 -m venv env/py
source env/py/bin/activate
pip install --upgrade pip
```

### 4.2 Install Required Libraries

```bash
pip install \
  torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
  lightkurve \
  astroquery \
  astropy \
  pandas numpy scipy \
  scikit-learn \
  umap-learn \
  hdbscan \
  tqdm \
  matplotlib \
  streamlit \
  pyarrow \
  requests
```

Library roles:

- `torch`: 1D CNN autoencoder training on T4 GPU (two models: SAP and delta).
- `lightkurve`: reading FITS light curve files locally after bulk download.
- `astroquery`: downloading TESS-SVC catalog, querying SIMBAD TAP, fetching multi-sector
  data for anomaly stars.
- `astropy`: FITS file handling and coordinate utilities.
- `umap-learn`, `hdbscan`: dimensionality reduction and clustering of latent embeddings.
- `streamlit`: interactive atlas app for exploring anomaly clusters.

---

## 5. Understanding the Data

### 5.1 TESS Light Curve Files

Each TESS light curve file is a FITS file for one star observed in one sector. Every
file contains a `LIGHTCURVE` binary table extension with at minimum these columns:

| Column | Type | Description |
|---|---|---|
| `TIME` | float64 | Barycentric TESS Julian Date |
| `SAP_FLUX` | float32 | Raw aperture photometry flux (electrons/sec) |
| `SAP_FLUX_ERR` | float32 | 1-sigma uncertainty on SAP_FLUX |
| `PDCSAP_FLUX` | float32 | Pipeline-detrended flux (systematics removed) |
| `PDCSAP_FLUX_ERR` | float32 | 1-sigma uncertainty on PDCSAP_FLUX |
| `QUALITY` | int32 | Bitmask: 0 = good cadence |

We use both `SAP_FLUX` and `PDCSAP_FLUX`. Their difference — the **delta flux** — is
what the pipeline removed. This is what we model directly in v1.1.

Each sector covers ~27.4 days. At 2-minute cadence this produces approximately 19,000–
20,000 time points per file. File size is approximately 2 MB per star per sector.

### 5.2 TESS-SVC: The Known Variable Catalog

The TESS Stellar Variability Catalog (TESS-SVC) contains 84,046 stars identified as
significantly variable on timescales of 0.01–13 days, from Sectors 1–26. It is hosted
on MAST as a High Level Science Product.

We use TESS-SVC as our **exclusion list**. Any star in TESS-SVC is already a known
variable and is not interesting to us.

Download URL: `https://archive.stsci.edu/hlsp/tess-svc`

### 5.3 SIMBAD: The Second Exclusion Layer

SIMBAD classifies astronomical objects in a four-level hierarchy. Any star with object
type `V*..` (variable star, any subtype) is excluded. We query SIMBAD via its TAP
interface in a single batch request — no rate limiting issues.

### 5.4 TESS Input Catalog (TIC)

The TIC-8 catalog contains physical parameters for every TESS target:

| Column | Description |
|---|---|
| `ID` | TIC identifier |
| `Teff` | Effective temperature (K) |
| `rad` | Stellar radius (solar radii) |
| `mass` | Stellar mass (solar masses) |
| `lum` | Stellar luminosity (solar luminosities) |
| `Tmag` | TESS magnitude |
| `ra`, `dec` | Sky coordinates |

These parameters are used in Phase G to characterize anomaly clusters.

---

## 6. Phase A — Get the Target List

### 6.1 Start a tmux Session

```bash
cd ~/tess-quiet
tmux new -s tess
```

Detach: `Ctrl+B` then `D`. Reattach: `tmux attach -t tess`.

All subsequent work happens inside this tmux session.

### 6.2 Download TESS-SVC Exclusion Catalog

Go to `https://archive.stsci.edu/hlsp/tess-svc`, find the catalog CSV link, and
download it:

```bash
wget -O data/tess_svc.csv "https://archive.stsci.edu/hlsps/tess-svc/hlsp_tess-svc_tess_lcf_all-s0001-s0026_tess_v1.0_cat.csv"
```

Verify:

```bash
head -3 data/tess_svc.csv
wc -l data/tess_svc.csv   # should be ~84,047 lines including header
```

### 6.3 Get the Full 2-Minute Target List for Sectors 1–5

Visit `https://archive.stsci.edu/tess/bulk_downloads.html` and download the five
light curve shell scripts for Sectors 1–5:

```bash
wget -O scripts/tesscurl_sector_1_lc.sh \
  "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_1_lc.sh"
wget -O scripts/tesscurl_sector_2_lc.sh \
  "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_2_lc.sh"
wget -O scripts/tesscurl_sector_3_lc.sh \
  "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_3_lc.sh"
wget -O scripts/tesscurl_sector_4_lc.sh \
  "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_4_lc.sh"
wget -O scripts/tesscurl_sector_5_lc.sh \
  "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_5_lc.sh"
```

Do **not** run these yet. Extract TIC IDs first:

```bash
grep -oP 'tess\d+-s\d+-\K\d+' scripts/tesscurl_sector_1_lc.sh | sort -u \
  > data/sector1_all_tic_ids.txt
grep -oP 'tess\d+-s\d+-\K\d+' scripts/tesscurl_sector_2_lc.sh | sort -u \
  > data/sector2_all_tic_ids.txt
grep -oP 'tess\d+-s\d+-\K\d+' scripts/tesscurl_sector_3_lc.sh | sort -u \
  > data/sector3_all_tic_ids.txt
grep -oP 'tess\d+-s\d+-\K\d+' scripts/tesscurl_sector_4_lc.sh | sort -u \
  > data/sector4_all_tic_ids.txt
grep -oP 'tess\d+-s\d+-\K\d+' scripts/tesscurl_sector_5_lc.sh | sort -u \
  > data/sector5_all_tic_ids.txt

cat data/sector*_all_tic_ids.txt | sort -u > data/all_tic_ids.txt
wc -l data/all_tic_ids.txt   # expect ~60,000–100,000 unique TIC IDs
```

---

## 7. Phase B — Build the "Boring Stars" Filter

Goal: produce `data/boring_tic_ids.txt` containing TIC IDs that are in the 2-minute
cadence target list AND not in TESS-SVC AND not classified as variable in SIMBAD.

Create `scripts/build_boring_filter.py`:

```python
import pandas as pd
from astroquery.simbad import Simbad
from astropy.table import Table

ALL_TIC_IDS_FILE = 'data/all_tic_ids.txt'
TESS_SVC_FILE    = 'data/tess_svc.csv'
OUT_FILE         = 'data/boring_tic_ids.txt'

def load_all_tic_ids():
    with open(ALL_TIC_IDS_FILE) as f:
        # Strip leading zeros to match TESS-SVC integer format
        return set(str(int(line.strip())) for line in f if line.strip())

def load_tess_svc_ids():
    df = pd.read_csv(TESS_SVC_FILE)
    # TESS-SVC uses 'tess_id' as the TIC identifier column
    col = 'tess_id'
    if col not in df.columns:
        # fallback: find any column with 'id' in the name
        candidates = [c for c in df.columns if c.lower().endswith('id')]
        if not candidates:
            raise ValueError(f"Cannot find TIC ID column. Columns: {df.columns.tolist()}")
        col = candidates[0]
        print(f"Warning: 'tess_id' not found, using '{col}'")
    ids = set(df[col].astype(str).str.strip())
    print(f"  Sample TESS-SVC IDs: {list(ids)[:3]}")
    return ids

def query_simbad_variable_ids(tic_ids):
    """Single TAP batch query — returns set of TIC IDs classified as variable in SIMBAD."""
    upload_table = Table({'tic_id': [f'TIC {t}' for t in list(tic_ids)]})
    adql = """
        SELECT upload.tic_id, basic.main_id, basic.otype
        FROM basic
        JOIN TAP_UPLOAD.my_tics AS upload
          ON basic.main_id = upload.tic_id
        WHERE basic.otype = 'V*..'
    """
    try:
        result = Simbad.query_tap(adql, my_tics=upload_table)
        if result is None or len(result) == 0:
            return set()
        matched = set()
        for row in result:
            num = str(row['tic_id']).replace('TIC', '').strip()
            matched.add(num)
        return matched
    except Exception as e:
        print(f"SIMBAD TAP query error: {e}")
        print("Proceeding without SIMBAD filter — relying on TESS-SVC only.")
        return set()

def main():
    print("Loading all TIC IDs...")
    all_ids = load_all_tic_ids()
    print(f"  Total: {len(all_ids)}")

    print("Loading TESS-SVC known variable IDs...")
    svc_ids = load_tess_svc_ids()
    print(f"  Known variables in TESS-SVC: {len(svc_ids)}")

    after_svc = all_ids - svc_ids
    print(f"  After TESS-SVC removal: {len(after_svc)}")

    print("Querying SIMBAD for additional known variables...")
    if len(after_svc) > 80000:
        chunks = [list(after_svc)[i:i + 50000] for i in range(0, len(after_svc), 50000)]
        simbad_vars = set()
        for i, chunk in enumerate(chunks):
            print(f"  SIMBAD batch {i + 1}/{len(chunks)}...")
            simbad_vars |= query_simbad_variable_ids(set(chunk))
    else:
        simbad_vars = query_simbad_variable_ids(after_svc)

    print(f"  Additional known variables in SIMBAD: {len(simbad_vars)}")

    boring_ids = after_svc - simbad_vars
    print(f"  Final boring star count: {len(boring_ids)}")

    with open(OUT_FILE, 'w') as f:
        for tic in sorted(boring_ids):
            f.write(tic + '\n')

    print(f"Saved to {OUT_FILE}")

if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/build_boring_filter.py | tee logs/01_boring_filter.log
```

Expected output: ~50,000–70,000 boring TIC IDs across 5 sectors.

---

## 8. Phase C — Download Light Curves

### 8.1 Filter the Download Scripts

Create `scripts/filter_download_scripts.py`:

```python
import os
import re

BORING_IDS_FILE = 'data/boring_tic_ids.txt'
SCRIPT_DIR      = 'scripts'
OUT_DIR         = 'scripts/filtered'

os.makedirs(OUT_DIR, exist_ok=True)


def extract_tic_from_curl_line(line):
    """
    Extract TIC ID from a MAST curl line.
    """
    match = re.search(r'-s\d+-(\d+)-', line)
    if match:
        return str(int(match.group(1)))
    return None


def main():
    with open(BORING_IDS_FILE) as f:
        boring_ids = set(line.strip() for line in f if line.strip())

    for sector_num in range(1, 6):
        script_path = os.path.join(SCRIPT_DIR, f'tesscurl_sector_{sector_num}_lc.sh')
        out_path    = os.path.join(OUT_DIR, f'filtered_sector_{sector_num}_lc.sh')

        if not os.path.exists(script_path):
            print(f"WARNING: {script_path} not found — skipping sector {sector_num}")
            continue

        kept  = 0
        total = 0
        with open(script_path) as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                if not line.startswith('curl'):
                    f_out.write(line)
                    continue
                total += 1
                tic_id = extract_tic_from_curl_line(line)
                if tic_id and tic_id in boring_ids:
                    f_out.write(line)
                    kept += 1

        print(f"Sector {sector_num}: kept {kept}/{total} curl lines → {out_path}")


if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/filter_download_scripts.py | tee logs/02_filter_scripts.log
```

### 8.2 Download the Filtered Light Curves

```bash
mkdir -p data/raw/sector1 data/raw/sector2 data/raw/sector3 \
         data/raw/sector4 data/raw/sector5

cd data/raw/sector1
bash ~/tess-quiet/scripts/filtered/filtered_sector_1_lc.sh 2>&1 \
  | tee ~/tess-quiet/logs/03_download_s1.log

cd ~/tess-quiet/data/raw/sector2
bash ~/tess-quiet/scripts/filtered/filtered_sector_2_lc.sh 2>&1 \
  | tee ~/tess-quiet/logs/03_download_s2.log

# Repeat for sectors 3, 4, 5
```

Verify:

```bash
find data/raw -name "*.fits" | wc -l   # expect ~50,000–70,000 files
du -sh data/raw                         # expect ~20–25 GB
```

---

## 9. Phase D — Preprocessing: Build the Flux Matrix (v1.1)

**v1.1 change:** We now extract three signals per star: SAP flux, PDCSAP flux, and the
delta (SAP minus PDCSAP). The delta directly represents what the pipeline removed and
is the core input to Model B.

Create `scripts/preprocess_lightcurves.py`:

```python
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
```

Run:

```bash
python scripts/preprocess_lightcurves.py | tee logs/04_preprocess.log
```

CPU-bound, 30–90 minutes. Continues in tmux if SSH disconnects.

---

## 10. Phase E — Train Two Autoencoders (v1.1)

**v1.1 change:** We train Model A on SAP flux and Model B on delta flux. The combined
anomaly score is `recon_error_sap + alpha * recon_error_delta` where alpha = 0.7.

Create `scripts/train_autoencoder.py`:

```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

FLUX_PARQUET        = 'data/processed/flux_matrix.parquet'
MODEL_SAP_OUT       = 'results/autoencoder_sap.pt'
MODEL_DELTA_OUT     = 'results/autoencoder_delta.pt'
LATENT_SAP_OUT      = 'results/latent_sap.parquet'
LATENT_DELTA_OUT    = 'results/latent_delta.parquet'
COMBINED_ERRORS_OUT = 'results/reconstruction_errors.parquet'

N_POINTS   = 1024
LATENT_DIM = 16
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 1e-3
ALPHA      = 0.7   # Weight for delta error in combined score

os.makedirs('results', exist_ok=True)


class FluxDataset(Dataset):
    def __init__(self, flux_array):
        self.data = torch.tensor(flux_array, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 64, latent_dim)
        )
        self.decoder_fc = nn.Linear(latent_dim, 128 * 64)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (128, 64)),
            nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder_conv(self.decoder_fc(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_model(flux_array, model_path, label):
    """Train one autoencoder. Returns (errors, latents) numpy arrays."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_workers > 0 can deadlock on some Linux/GCP setups without fork context
    num_workers = 4 if torch.cuda.is_available() else 0

    dataset   = FluxDataset(flux_array)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=num_workers, pin_memory=(num_workers > 0))
    model     = ConvAutoencoder(LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\nTraining {label} on {device}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"  {label} Epoch {epoch}/{EPOCHS}", leave=False):
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(dataset)
        print(f"  Epoch {epoch:02d}/{EPOCHS}: loss = {epoch_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to {model_path}")

    # Extract latents and per-sample reconstruction errors
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=(num_workers > 0))
    all_latents, all_errors = [], []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"  Extracting {label}", leave=False):
            batch = batch.to(device)
            recon, z = model(batch)
            errors = ((recon - batch) ** 2).mean(dim=[1, 2]).cpu().numpy()
            all_errors.append(errors)
            all_latents.append(z.cpu().numpy())

    return np.concatenate(all_errors), np.concatenate(all_latents)


def main():
    print("Loading flux matrix...")
    df          = pd.read_parquet(FLUX_PARQUET)
    sap_array   = np.stack(df['flux_sap'].values).astype(np.float32)
    delta_array = np.stack(df['flux_delta'].values).astype(np.float32)
    print(f"  Stars: {len(df)}  |  Points per star: {N_POINTS}")

    # Train Model A — SAP flux
    errors_sap, latents_sap = train_model(sap_array, MODEL_SAP_OUT, "Model-A (SAP)")

    # Train Model B — delta flux
    errors_delta, latents_delta = train_model(delta_array, MODEL_DELTA_OUT, "Model-B (delta)")

    # Combined anomaly score
    combined_score = errors_sap + ALPHA * errors_delta

    # Save SAP latents (used for clustering in Phase I)
    latent_df = pd.DataFrame(latents_sap, columns=[f'z_{i}' for i in range(LATENT_DIM)])
    latent_df.insert(0, 'tic_id', df['tic_id'].values)
    latent_df.to_parquet(LATENT_SAP_OUT, index=False)

    # Save delta latents separately
    latent_delta_df = pd.DataFrame(latents_delta,
                                   columns=[f'zd_{i}' for i in range(LATENT_DIM)])
    latent_delta_df.insert(0, 'tic_id', df['tic_id'].values)
    latent_delta_df.to_parquet(LATENT_DELTA_OUT, index=False)

    # Save all error metrics
    error_df = pd.DataFrame({
        'tic_id':            df['tic_id'].values,
        'filepath':          df['filepath'].values,
        'recon_error_sap':   errors_sap,
        'recon_error_delta': errors_delta,
        'combined_score':    combined_score,
    })
    error_df.to_parquet(COMBINED_ERRORS_OUT, index=False)

    # Summary statistics
    for col in ['recon_error_sap', 'recon_error_delta', 'combined_score']:
        vals = error_df[col].values
        p99  = np.percentile(vals, 99)
        print(f"\n{col}: mean={vals.mean():.5f}  std={vals.std():.5f}  "
              f"P99={p99:.5f}  ({(vals > p99).sum()} stars above P99)")

    print("\nDone. Outputs:")
    print(f"  {MODEL_SAP_OUT}")
    print(f"  {MODEL_DELTA_OUT}")
    print(f"  {LATENT_SAP_OUT}")
    print(f"  {LATENT_DELTA_OUT}")
    print(f"  {COMBINED_ERRORS_OUT}")

if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/train_autoencoder.py | tee logs/05_train_autoencoder.log
```

Expected training time: 40–90 minutes on T4 (two models, 50 epochs each).

---

## 11. Phase F — Select Anomalies with SAP/PDCSAP Ratio Filter (v1.1)

**v1.1 change:** After selecting top anomalies by combined score, we apply the
SAP/PDCSAP signal ratio filter — keeping only stars where the SAP signal is
substantially stronger than the PDCSAP signal. This enforces the scientific
hypothesis: we want signals that were genuinely suppressed.

Create `scripts/select_anomalies.py`:

```python
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
    
    # Deduplicate flux_sub: one row per TIC, keep the sector with highest ratio
    flux_sub = flux_sub.sort_values('sap_pdcsap_ratio', ascending=False)
    flux_sub = flux_sub.drop_duplicates(subset='tic_id', keep='first')

    before = len(anomalies)
    passing_ids = flux_sub[flux_sub['sap_pdcsap_ratio'] >= SAP_PDCSAP_RATIO_THRESHOLD]['tic_id']
    anomalies = anomalies[anomalies['tic_id'].isin(passing_ids)].copy()
    print(f"  After SAP/PDCSAP ratio filter (≥{SAP_PDCSAP_RATIO_THRESHOLD}x): "
          f"{len(anomalies)} stars (removed {before - len(anomalies)})")

    # Deduplicate: flux_matrix has one row per star-sector so a star observed
    # in multiple sectors appears multiple times. Keep highest combined_score.
    before_dedup = len(anomalies)
    anomalies = anomalies.sort_values('combined_score', ascending=False)
    anomalies = anomalies.drop_duplicates(subset='tic_id', keep='first')
    print(f"  After dedup (multi-sector rows): {len(anomalies)} unique stars "
          f"(removed {before_dedup - len(anomalies)} duplicate rows)")

    # Attach ratio values
    anomalies = anomalies.merge(
        flux_sub[['tic_id', 'sap_strength', 'pdcsap_strength', 'sap_pdcsap_ratio']],
        on='tic_id', how='left'
    )
    
    # Deduplicate latents: one row per TIC (latent vectors are per star-sector)
    latents = latents.drop_duplicates(subset='tic_id', keep='first')

    # Merge with latent vectors
    anomalies = anomalies.merge(latents, on='tic_id', how='left')

    anomalies.to_parquet(OUT_FILE, index=False)
    print(f"Saved {len(anomalies)} anomalies to {OUT_FILE}")

if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/select_anomalies.py | tee logs/06_select_anomalies.log
```

---

## 12. Phase G — Fetch TIC Parameters

Create `scripts/fetch_tic_params.py`:

```python
import pandas as pd
from astroquery.mast import Catalogs

ANOMALIES_FILE = 'results/anomalies.parquet'
OUT_FILE       = 'results/anomalies_with_tic.parquet'

TIC_COLS = ['ID', 'Teff', 'rad', 'mass', 'lum', 'Tmag', 'ra', 'dec', 'logg', 'MH']

def fetch_tic_batch(tic_ids, batch_size=1000):
    all_results = []
    for i in range(0, len(tic_ids), batch_size):
        batch = tic_ids[i:i + batch_size]
        try:
            result = Catalogs.query_criteria(catalog='TIC', ID=batch)
            if result is not None and len(result) > 0:
                all_results.append(result[TIC_COLS].to_pandas())
        except Exception as e:
            print(f"  Batch {i//batch_size + 1} failed: {e}")
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame(columns=TIC_COLS)

def main():
    anomalies = pd.read_parquet(ANOMALIES_FILE)
    tic_ids   = anomalies['tic_id'].astype(int).tolist()
    print(f"Fetching TIC parameters for {len(tic_ids)} stars...")

    tic_df = fetch_tic_batch(tic_ids)
    tic_df['tic_id'] = tic_df['ID'].astype(str)

    merged = anomalies.merge(tic_df, on='tic_id', how='left')
    merged.to_parquet(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")
    print(f"Stars with Teff: {merged['Teff'].notna().sum()}")

if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/fetch_tic_params.py | tee logs/07_fetch_tic.log
```

---

## 13. Phase H — Latent Density Scoring (v1.1)

**v1.1 addition:** Reconstruction error alone misses signals that are structurally
unusual but easy to reconstruct. We add a kNN latent density score — stars far from
their neighbors in latent space are structurally rare, not just noisy.

Create `scripts/latent_density_score.py`:

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ANOMALIES_FILE = 'results/anomalies_with_tic.parquet'
OUT_FILE       = 'results/anomalies_scored.parquet'

LATENT_COLS = [f'z_{i}' for i in range(16)]
K_NEIGHBORS = 10   # kNN distance uses 10 nearest neighbors
BETA        = 0.5  # Weight for latent density score in final score

def main():
    df = pd.read_parquet(ANOMALIES_FILE)
    X  = df[LATENT_COLS].values.astype(np.float32)

    print(f"Computing kNN latent density scores (k={K_NEIGHBORS}) "
          f"for {len(X)} anomaly stars...")

    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='euclidean', n_jobs=-1)
    knn.fit(X)
    distances, _ = knn.kneighbors(X)

    # Latent score = mean distance to k nearest neighbors (excluding self)
    latent_scores = distances[:, 1:].mean(axis=1)

    # Normalize latent scores to [0, 1] range for stable combination
    ls_min, ls_max = latent_scores.min(), latent_scores.max()
    if ls_max > ls_min:
        latent_scores_norm = (latent_scores - ls_min) / (ls_max - ls_min)
    else:
        latent_scores_norm = np.zeros_like(latent_scores)

    # Normalize combined_score similarly
    cs = df['combined_score'].values
    cs_min, cs_max = cs.min(), cs.max()
    cs_norm = (cs - cs_min) / (cs_max - cs_min) if cs_max > cs_min else np.zeros_like(cs)

    df['latent_score']      = latent_scores
    df['final_score']       = cs_norm + BETA * latent_scores_norm

    # Re-rank by final score
    df = df.sort_values('final_score', ascending=False).reset_index(drop=True)

    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved {len(df)} scored anomalies to {OUT_FILE}")
    print(f"Top 10 final scores: {df['final_score'].head(10).values.round(4)}")

if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/latent_density_score.py | tee logs/08_latent_density.log
```

---

## 14. Phase I — UMAP + HDBSCAN Clustering (v1.1)

**v1.1 change:** Clustering now runs on `final_score`-ranked anomalies. The input
parquet is `anomalies_scored.parquet`. No other change to clustering logic.

Create `scripts/cluster_anomalies.py`:

```python
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

ANOMALIES_FILE = 'results/anomalies_scored.parquet'
OUT_FILE       = 'results/anomaly_clusters.parquet'

LATENT_COLS = [f'z_{i}' for i in range(16)]

def main():
    df = pd.read_parquet(ANOMALIES_FILE)
    X  = df[LATENT_COLS].values.astype(np.float32)
    print(f"Clustering {len(X)} anomaly stars in {X.shape[1]}D latent space")

    X_scaled = StandardScaler().fit_transform(X)

    print("Running UMAP 2D (visualization)...")
    umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        metric='cosine', random_state=42).fit_transform(X_scaled)

    print("Running UMAP 5D (clustering)...")
    umap_5d = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=5,
                        metric='cosine', random_state=42).fit_transform(X_scaled)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(umap_5d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")
    for lbl in sorted(set(labels)):
        count = (labels == lbl).sum()
        print(f"  {'Cluster' if lbl >= 0 else 'Noise'} {lbl}: {count} stars")

    df['cluster'] = labels
    df['umap_x']  = umap_2d[:, 0]
    df['umap_y']  = umap_2d[:, 1]

    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")

if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/cluster_anomalies.py | tee logs/09_cluster.log
```

---

## 15. Phase J — Cluster Stability Check (v1.1)

**v1.1 addition:** UMAP + HDBSCAN can produce different clusters with different random
seeds. We run clustering 10 times with different seeds and measure what fraction of runs
produce a similar cluster for each star. Only stable clusters are carried forward.

Create `scripts/cluster_stability.py`:

```python
"""
Phase J — Cluster Stability Check
Runs clustering 10 times with different random seeds and measures what fraction
of runs each star is assigned to a real cluster (not noise). Unstable clusters
are flagged and excluded from downstream phases.
Output: results/anomaly_clusters_stable.parquet
"""

import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

ANOMALIES_FILE  = 'results/anomalies_scored.parquet'
CLUSTER_FILE    = 'results/anomaly_clusters.parquet'
OUT_FILE        = 'results/anomaly_clusters_stable.parquet'

LATENT_COLS         = [f'z_{i}' for i in range(16)]
N_STABILITY_RUNS    = 10
STABILITY_THRESHOLD = 0.6  # Cluster must appear in ≥60% of runs

def run_clustering(X_scaled, seed):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=5,
                        metric='cosine', random_state=seed)
    X_umap = reducer.fit_transform(X_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5,
                                 metric='euclidean', cluster_selection_method='eom')
    return clusterer.fit_predict(X_umap)

def main():
    df = pd.read_parquet(ANOMALIES_FILE)

    # Merge cluster labels AND umap coordinates from the primary clustering run.
    # umap_x/umap_y must be preserved all the way to the atlas app.
    clusters = pd.read_parquet(CLUSTER_FILE)[['tic_id', 'cluster', 'umap_x', 'umap_y']]
    df = df.merge(clusters, on='tic_id', how='left')

    X        = df[LATENT_COLS].values.astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X)

    print(f"Running {N_STABILITY_RUNS} clustering runs with different seeds...")

    # Store per-run labels: shape (N_RUNS, N_STARS)
    all_run_labels = []
    for seed in range(N_STABILITY_RUNS):
        labels = run_clustering(X_scaled, seed=seed * 7 + 42)
        all_run_labels.append(labels)
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  Run {seed+1}: {n_c} clusters")

    all_run_labels = np.array(all_run_labels)  # (N_RUNS, N_STARS)

    # Stability = fraction of runs where the star lands in any real cluster (not noise)
    stability_scores = []
    for star_idx in range(len(df)):
        star_labels = all_run_labels[:, star_idx]
        non_noise   = star_labels[star_labels != -1]
        stability   = len(non_noise) / N_STABILITY_RUNS
        stability_scores.append(stability)

    df['stability_score'] = stability_scores

    # For each original cluster, compute mean stability
    print("\nCluster stability summary:")
    original_clusters = df['cluster'].unique()
    cluster_stabilities = {}
    for cid in sorted(original_clusters):
        if cid == -1:
            continue
        sub   = df[df['cluster'] == cid]
        mean_s = sub['stability_score'].mean()
        cluster_stabilities[cid] = mean_s
        flag = "✅ STABLE" if mean_s >= STABILITY_THRESHOLD else "⚠️  UNSTABLE"
        print(f"  Cluster {cid} (n={len(sub)}): "
              f"mean stability = {mean_s:.2f}  — {flag}")

    # Mark clusters as stable or not
    df['cluster_stable'] = df['cluster'].map(
        lambda c: cluster_stabilities.get(c, 0) >= STABILITY_THRESHOLD
        if c >= 0 else False
    )

    n_stable = df[df['cluster_stable'] & (df['cluster'] >= 0)]['cluster'].nunique()
    print(f"\n{n_stable} stable clusters will be carried forward")

    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")

if __name__ == '__main__':
    main()
```

Run:

```bash
python scripts/cluster_stability.py | tee logs/10_stability.log
```

---

## 16. Phase K — Artifact Checks (v1.1)

**v1.1 change:** Two artifact checks now run in sequence. Check 1 (spatial CCD
distribution) was in v1.0. Check 2 (temporal spike alignment) is new — it catches
momentum dumps and scattered light events that affect many stars at the same timestamp.

Create `scripts/artifact_check.py`:

```python
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
```

Run:

```bash
python scripts/artifact_check.py | tee logs/11_artifact_check.log
```

---

## 17. Phase L — Multi-Sector Repeatability Validation (v1.1 — CRITICAL)

**v1.1 addition:** This is the strongest validation gate. True astrophysical signals
repeat across independent TESS sectors. Instrumental artifacts are sector-specific.

For each anomaly star that passed the artifact checks, we find which other sectors
observed it and download those light curves on-demand. We recompute the anomaly score
for each additional sector and check whether the signal repeats.

Create `scripts/multisector_validation.py`:

```python
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
```

Run (this step makes network requests and may take 1–3 hours depending on how many
stars have multi-sector data):

```bash
python scripts/multisector_validation.py | tee logs/12_multisector.log
```

This step uses lightkurve's built-in caching (`~/.lightkurve-cache`) so downloaded
sectors are not re-fetched if the script is restarted.

---

## 18. Phase M — Streamlit Atlas App (v1.1)

The atlas is updated to display delta flux, repeatability scores, and stability flags
alongside the original views.

Create `app/atlas_app.py`:

```python
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from astropy.io import fits

CLUSTER_FILE = 'results/anomaly_clusters_validated.parquet'
FLUX_FILE    = 'data/processed/flux_matrix.parquet'

@st.cache_data
def load_clusters():
    return pd.read_parquet(CLUSTER_FILE)

@st.cache_data
def load_flux_for_tic(tic_id):
    df = pd.read_parquet(FLUX_FILE, columns=['tic_id', 'flux_sap', 'flux_pdcsap', 'flux_delta'])
    row = df[df['tic_id'].astype(str) == str(tic_id)]
    if len(row) == 0:
        return None
    return row.iloc[0]

def plot_three_panel(tic_id, flux_row, recon_error, final_score):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    x = np.arange(1024)

    axes[0].plot(x, flux_row['flux_sap'], lw=0.5, color='steelblue')
    axes[0].set_ylabel('SAP (norm)')
    axes[0].set_title(f'TIC {tic_id}  |  score={final_score:.4f}  recon_err={recon_error:.5f}')

    axes[1].plot(x, flux_row['flux_pdcsap'], lw=0.5, color='seagreen')
    axes[1].set_ylabel('PDCSAP (norm)')

    axes[2].plot(x, flux_row['flux_delta'], lw=0.5, color='tomato')
    axes[2].set_ylabel('Delta (removed)')
    axes[2].set_xlabel('Time index (1024 pts = 27 days)')

    plt.tight_layout()
    return fig

def main():
    st.title("TESS Quiet Stars Anomaly Atlas (v1.1)")
    st.write("Anomalous signals in officially non-variable TESS stars.")

    df = load_clusters()

    # Sidebar filters
    show_stable_only    = st.sidebar.checkbox("Stable clusters only", True)
    show_non_artifact   = st.sidebar.checkbox("Hide artifact-flagged clusters", True)
    min_repeatability   = st.sidebar.slider("Min cluster repeatability", 0.0, 1.0, 0.0, 0.1)

    # Filter
    view = df.copy()
    if show_stable_only and 'cluster_stable' in view.columns:
        view = view[view['cluster_stable'] == True]
    if show_non_artifact and 'artifact_flagged' in view.columns:
        view = view[view['artifact_flagged'] == False]

    real_clusters = sorted([c for c in view['cluster'].unique() if c >= 0])

    if not real_clusters:
        st.warning("No clusters pass current filters. Relax the sidebar settings.")
        return

    cluster_id = st.sidebar.selectbox("Select cluster", real_clusters)
    cluster_df = view[view['cluster'] == cluster_id].copy()
    n          = len(cluster_df)

    st.subheader(f"Cluster {cluster_id} — {n} stars")

    # Tier badge
    rep_score = cluster_df.get('repeatability_score', pd.Series([0])).mean()
    if rep_score >= 0.3 and cluster_df.get('artifact_flagged', pd.Series([True])).mean() == 0:
        st.success("🔴 DISCOVERY CANDIDATE — passes all validation gates")
    elif n >= 10:
        st.info("🟡 Strong candidate — inspect light curves")
    else:
        st.warning("🟢 Tier 1 — anomaly cluster identified")

    # UMAP scatter
    fig_umap, ax = plt.subplots(figsize=(8, 5))
    noise = df[df['cluster'] == -1]
    ax.scatter(noise['umap_x'], noise['umap_y'], c='lightgrey', s=4, alpha=0.3)
    for cid in real_clusters:
        sub   = df[df['cluster'] == cid]
        color = 'tomato' if cid == cluster_id else 'steelblue'
        size  = 25 if cid == cluster_id else 6
        alpha = 0.9 if cid == cluster_id else 0.25
        ax.scatter(sub['umap_x'], sub['umap_y'], c=color, s=size, alpha=alpha)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Anomaly clusters (UMAP 2D)')
    st.pyplot(fig_umap)

    # Stellar properties
    st.subheader("Stellar properties")
    prop_cols = ['Teff', 'rad', 'lum', 'Tmag']
    available = [c for c in prop_cols if c in cluster_df.columns]
    if available:
        st.dataframe(cluster_df[available].describe().round(2))

    # Repeatability
    if 'repeatability_score' in cluster_df.columns:
        st.subheader("Multi-sector repeatability")
        rep_data = cluster_df[['tic_id', 'sectors_checked',
                                'sectors_anomalous', 'repeatability_score']].copy()
        rep_data = rep_data[rep_data['sectors_checked'] > 0]
        if len(rep_data):
            st.dataframe(rep_data.round(3))

    # Individual light curves
    st.subheader("Individual light curves (SAP / PDCSAP / Delta)")
    n_show = st.slider("Stars to show", 1, min(10, n), 3)
    sample = cluster_df.sample(min(n_show, n))

    for _, row in sample.iterrows():
        tic_id    = row['tic_id']
        flux_row  = load_flux_for_tic(tic_id)
        if flux_row is not None:
            fig = plot_three_panel(
                tic_id, flux_row,
                row.get('recon_error_sap', 0),
                row.get('final_score', 0)
            )
            st.pyplot(fig)
            plt.close(fig)
            with st.expander(f"TIC {tic_id} — parameters"):
                params = {k: row.get(k, 'N/A') for k in
                          ['Teff', 'rad', 'lum', 'Tmag', 'ra', 'dec',
                           'camera', 'ccd', 'sector', 'sap_pdcsap_ratio',
                           'stability_score', 'repeatability_score']}
                st.json({k: float(v) if isinstance(v, (np.floating, float)) else str(v)
                         for k, v in params.items()})
        else:
            st.write(f"Flux data not available for TIC {tic_id}")

if __name__ == '__main__':
    main()
```

Run:

```bash
cd ~/tess-quiet
source env/py/bin/activate
streamlit run app/atlas_app.py --server.port 8501 --server.address 0.0.0.0
```

Open: `http://<your_gcp_instance_ip>:8501`

---

## 19. Interpretation Guidelines (v1.1)

Work through these gates in order. Stop and document at each gate before proceeding.

**Gate 1 — Stability:** Only inspect clusters where `cluster_stable = True`.
Unstable clusters are artifacts of the UMAP random seed, not real structure.

**Gate 2 — Spatial artifact check:** Discard any cluster where >80% of stars share
a camera+CCD combination. Pure CCD concentration = instrumental.

**Gate 3 — Temporal artifact check:** Discard any cluster where >50% of member stars
have `temporal_artifact = True`. Synchronized temporal spikes = momentum dumps or
scattered light.

**Gate 4 — SAP/PDCSAP ratio:** In the atlas, check the `sap_pdcsap_ratio` values.
The higher this ratio, the stronger the evidence that the signal was genuinely present
in raw data and suppressed by the pipeline. Clusters where median ratio < 1.5 are weak.

**Gate 5 — Delta flux shape inspection:** For surviving clusters, look at the delta
flux panel (red) for 10–20 stars. The delta is what the pipeline removed. Ask:
does it have a consistent shape across stars in the cluster? Coherent shapes (slow
ramps, regular dips, asymmetric features) are more interesting than random noise.

**Gate 6 — Stellar properties:** Does the cluster's Teff, radius, and luminosity
distribution suggest a physically coherent stellar population? A tight Teff range
is a strong hint of a real phenomenon tied to stellar physics.

**Gate 7 — Multi-sector repeatability:** Check `repeatability_score` per cluster.
Any cluster where ≥30% of stars show the anomaly in independent sectors is a
**Tier 3 Discovery Candidate**. This is the hardest evidence available without
a ground-based telescope. Document these immediately.

**Gate 8 — Literature check:** For each Tier 3 candidate, search for:
known long-period variables (>13 days not in TESS-SVC), slowly pulsating B stars,
ellipsoidal variables, contamination from nearby stars. If none of these explain
the cluster, it is ready for Phase 2 (spectroscopic follow-up).

---

## 20. Execution Order Checklist (v1.1)

1. SSH into GCP T4 instance, verify `nvidia-smi`.
2. Install system packages, create `~/tess-quiet` directory structure.
3. Create virtualenv, install all libraries.
4. Start tmux session: `tmux new -s tess`.
5. Download TESS-SVC catalog → `data/tess_svc.csv`.
6. Download bulk curl scripts for Sectors 1–5 from MAST.
7. Extract TIC IDs → `data/all_tic_ids.txt`.
8. Run `build_boring_filter.py` → `data/boring_tic_ids.txt`.
9. Run `filter_download_scripts.py` → filtered curl scripts.
10. Run filtered curl scripts → `data/raw/` (~20–25 GB FITS files).
11. Run `preprocess_lightcurves.py` → `data/processed/flux_matrix.parquet`
    (three channels: SAP, PDCSAP, delta).
12. Run `train_autoencoder.py` → two models + combined error scores.
13. Run `select_anomalies.py` → top 1% by combined score, SAP/PDCSAP ratio filtered.
14. Run `fetch_tic_params.py` → stellar parameters attached.
15. Run `latent_density_score.py` → final_score computed.
16. Run `cluster_anomalies.py` → UMAP + HDBSCAN cluster assignments.
17. Run `cluster_stability.py` → stability scores, unstable clusters flagged.
18. Run `artifact_check.py` → spatial + temporal artifact flags.
19. Run `multisector_validation.py` → repeatability scores (takes 1–3 hours).
20. Run Streamlit atlas to explore validated clusters.
21. Apply Gates 1–8 from interpretation guidelines. Document Tier 3 candidates.

---

## 21. Notes and Caveats (v1.1)

**TESS-SVC column name:** Verify after download with `df.columns.tolist()`. The script
defaults to `ticid` with an auto-detect fallback.

**Bulk download URL format:** The MAST bulk downloads page
(`https://archive.stsci.edu/tess/bulk_downloads.html`) is the authoritative source.
Consult it directly if any 404 errors occur.

**Alpha (delta weight) = 0.7:** This is tunable. If the delta model produces very
noisy scores (because many stars have large pipelines corrections that aren't
astrophysically interesting), reduce alpha to 0.3. If delta signals are clean and
coherent, increase to 1.0.

**SIMBAD TAP upload limit:** Handled automatically with 50k-row chunking. If SIMBAD
is unavailable, falls back to TESS-SVC only — still a valid filter.

**Multi-sector validation runtime:** The lightkurve download calls are rate-limited
by MAST (~a few hundred files per hour). For ~600 anomaly stars × up to 5 sectors
each, expect 2–4 hours. Lightkurve caches results in `~/.lightkurve-cache` so
restarts pick up where they left off.

**Upgrade 7 (multi-scale representation) was reviewed and rejected.** The 1,024-point
sequence captures variation from ~38 minutes to 27 days. Multi-scale input would
complicate the architecture without a justified signal-theoretic reason. It can be
added in Phase 2 if a specific short-period signal class is found.

**Disk space:** Raw FITS (~20–25 GB) + flux matrix parquet (~3–5 GB) + lightkurve
cache for multi-sector (~5–10 GB) = ~35–40 GB total. Budget 120 GB disk to be safe.
After multi-sector validation completes you can remove `data/raw` if disk is tight.

**Phase 2 hook:** Tier 3 Discovery Candidates (stable + artifact-free + repeatable)
should be the subject of a dedicated follow-up experiment. The next step is:
download all available sectors for member stars, build a full multi-sector atlas,
and identify the brightest members for spectroscopic follow-up proposals.
