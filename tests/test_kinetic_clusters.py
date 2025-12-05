import scvelo as scv
import scanpy as sc
import pytest

def test_kinetic_clusters():
    adata = scv.datasets.pancreas()
    sc.pp.filter_genes(adata, min_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=500) # Smaller subset for testing
    scv.pp.moments(adata, n_pcs=10, n_neighbors=10)

    scv.tl.recover_dynamics(adata, max_iter=10, n_jobs=1) 

    scv.tl.kinetic_clusters(adata, n_clusters=3)
    
    assert 'kinetic_cluster' in adata.var.keys()
    assert adata.var['kinetic_cluster'].value_counts().sum() > 0

    try:
        scv.pl.kinetic_clusters(adata, show=False)
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")