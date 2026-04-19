# TESS Quiet Stars Moonshot

Discover anomalous stellar signals hidden inside officially "non-variable" stars from NASA's TESS mission.

We train two 1D convolutional autoencoders — one on raw SAP flux, one on delta flux (SAP minus PDCSAP, i.e. what the pipeline removed) — then cluster the high-anomaly stars by latent embedding, apply layered artifact and stability filtering, and validate with multi-sector repeatability.

Full experiment spec: `tess_quiet_stars_moonshot_spec_v1.1.md`

**Paper:** https://doi.org/10.5281/zenodo.19654856
**Dataset:** https://doi.org/10.5281/zenodo.19638863

---

## Infrastructure

- **GCP T4 instance** — all data downloads, preprocessing, training, validation
- **Local (Mac)** — code, git, iteration

---

## Setup (GCP)

```bash
git clone <repo-url>
cd tess-quiet

python3 -m venv env/py
source env/py/bin/activate

# PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Everything else
pip install -r requirements.txt
```

Start a tmux session before running any phase:

```bash
tmux new -s tess
source env/py/bin/activate
```

---

## Execution Order

| Phase | Script | Description |
|-------|--------|-------------|
| A | — | Download TESS-SVC catalog + bulk curl scripts, extract TIC IDs |
| B | `scripts/build_boring_filter.py` | Exclude known variables (TESS-SVC + SIMBAD) |
| C | `scripts/filter_download_scripts.py` + curl | Download filtered FITS light curves (~20–25 GB) |
| D | `scripts/preprocess_lightcurves.py` | Build flux matrix parquet (SAP / PDCSAP / delta) |
| E | `scripts/train_autoencoder.py` | Train two 1D CNN autoencoders on T4 GPU |
| F | `scripts/select_anomalies.py` | Top 1% by combined score + SAP/PDCSAP ratio filter |
| G | `scripts/fetch_tic_params.py` | Attach TIC stellar parameters |
| H | `scripts/latent_density_score.py` | kNN latent density scoring |
| I | `scripts/cluster_anomalies.py` | UMAP + HDBSCAN clustering |
| J | `scripts/cluster_stability.py` | 10-run stability check |
| K | `scripts/artifact_check.py` | Spatial CCD + temporal spike artifact filters |
| L | `scripts/multisector_validation.py` | Multi-sector repeatability validation |
| M | `app/atlas_app.py` | Streamlit interactive atlas |

Each phase logs to `logs/NN_<name>.log`.

---

## Success Tiers

- **Tier 1** — Anomaly clusters identified, artifact-filtered, visually inspectable
- **Tier 2** — Cluster with consistent shape + shared stellar properties + stable across runs
- **Tier 3 (Discovery Candidate)** — Tier 2 + high SAP/PDCSAP divergence + multi-sector repeatability in ≥30% of members

---

## Key Tunable Parameters

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| `ALPHA` | `train_autoencoder.py` | 0.7 | Weight of delta error in combined score |
| `ANOMALY_PERCENTILE` | `select_anomalies.py` | 99 | Top N% by combined score |
| `SAP_PDCSAP_RATIO_THRESHOLD` | `select_anomalies.py` | 1.5 | Min SAP/PDCSAP signal ratio |
| `N_STABILITY_RUNS` | `cluster_stability.py` | 10 | Clustering reruns for stability check |
| `STABILITY_THRESHOLD` | `cluster_stability.py` | 0.6 | Min fraction of runs in a cluster |
| `REPEATABILITY_MIN` | `multisector_validation.py` | 0.30 | Min fraction of sectors showing anomaly |
