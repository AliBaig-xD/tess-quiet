"""
Phase E — Train Two 1D CNN Autoencoders
Model A: SAP flux  |  Model B: delta flux (what the pipeline removed)
Combined anomaly score = recon_error_sap + ALPHA * recon_error_delta
Output: results/autoencoder_sap.pt, results/autoencoder_delta.pt,
        results/latent_sap.parquet, results/latent_delta.parquet,
        results/reconstruction_errors.parquet
"""

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
