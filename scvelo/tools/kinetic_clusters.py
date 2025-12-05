# scvelo/tools/kinetic_clusters.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .. import logging as logg

def kinetic_clusters(adata, n_clusters=4, copy=False):
    """
    Clustering of genes based on their kinetic parameters (alpha, beta, gamma).
    
    This identifies functional groups of genes (e.g. 'Fast Response', 'Transient', 
    'Accumulating') based on the differential equations learned by recover_dynamics.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_clusters
        Number of kinetic regimes to find.
    copy
        Return a copy instead of writing to adata.
        
    Returns
    -------
    Updates `adata.var` with the column `kinetic_cluster`.
    """
    logg.info('clustering genes by kinetics', r=True)

    # 1. Validation
    if 'fit_alpha' not in adata.var.keys():
        raise ValueError("Kinetic parameters not found. Run scv.tl.recover_dynamics(adata) first.")

    # 2. Extract Data (Only use genes that were successfully fitted)
    # We use R2 > 0 to filter out genes where the model failed
    valid_mask = (adata.var['fit_r2'] > 0) & (adata.var['fit_alpha'].notnull())
    
    if valid_mask.sum() < n_clusters:
        raise ValueError("Not enough fitted genes to perform clustering.")

    # Get the 3 parameters: Transcription (alpha), Splicing (beta), Degradation (gamma)
    features = adata.var.loc[valid_mask, ['fit_alpha', 'fit_beta', 'fit_gamma']]

    # 3. Log Transform & Scale
    # Biological rates span orders of magnitude, so we log-transform first.
    X = np.log1p(features.values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Clustering (KMeans)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # 5. Save Results
    # Initialize column with 'nan' strings
    col_name = 'kinetic_cluster'
    adata.var[col_name] = 'nan'
    # Fill in the clusters for the valid genes
    adata.var.loc[valid_mask, col_name] = clusters.astype(str)

    logg.info('    finished', time=True, end=' ')
    logg.hint(f'added \n    {col_name!r}, clusters of kinetic parameters (adata.var)')

    return adata if copy else None