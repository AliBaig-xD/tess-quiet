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