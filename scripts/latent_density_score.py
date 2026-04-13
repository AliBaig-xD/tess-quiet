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