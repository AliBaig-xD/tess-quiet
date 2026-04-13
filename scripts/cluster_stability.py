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