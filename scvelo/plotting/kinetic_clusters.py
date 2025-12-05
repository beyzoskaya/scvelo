# scvelo/plotting/kinetic_clusters.py
import matplotlib.pyplot as plt
import numpy as np
from .. import settings

def kinetic_clusters(adata, show=True, save=None):
    """
    Scatter plot of kinetic parameters colored by cluster.
    Plots Degradation (Gamma) vs Transcription (Alpha).
    """
    if 'kinetic_cluster' not in adata.var.keys():
        raise ValueError("Please run scv.tl.kinetic_clusters(adata) first.")

    # Extract data for plotting
    # We only plot genes that have a cluster assigned (not 'nan')
    mask = adata.var['kinetic_cluster'] != 'nan'
    df = adata.var.loc[mask].copy()
    
    clusters = df['kinetic_cluster'].unique()
    clusters = sorted(clusters) # Keep order consistent

    # Setup Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot each cluster with a different color
    for cluster_id in clusters:
        subset = df[df['kinetic_cluster'] == cluster_id]
        ax.scatter(
            subset['fit_gamma'], 
            subset['fit_alpha'], 
            label=f'Cluster {cluster_id}',
            s=15, alpha=0.6, edgecolors='none'
        )

    ax.set_xlabel('Degradation Rate (Gamma)')
    ax.set_ylabel('Transcription Rate (Alpha)')
    ax.set_title('Gene Kinetic Regimes')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    if save:
        plt.savefig(save, dpi=300)
    
    if show:
        plt.show()
    
    return ax