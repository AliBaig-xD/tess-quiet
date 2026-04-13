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