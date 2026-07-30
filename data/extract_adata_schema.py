import anndata as ad
import numpy as np
import scipy.sparse as sp
import json

def extract_agent_context(adata: ad.AnnData, max_categories: int = 20) -> str:
    """
    Extracts a context payload specifically designed for an LLM Planner agent.
    """
    context = {
        "dataset_shape": {
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars
        },
        "metadata_schema": {
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns)
        },
        "biological_context": {},
        "data_processing_state": {},
        "available_structures": {
            "obsm_embeddings": list(adata.obsm.keys()),
            "layers": list(adata.layers.keys()),
            "uns_keys": list(adata.uns.keys())
        }
    }

    # 1. Biological Metadata Values (Low-cardinality extraction)
    # We only extract unique values if the column is categorical/low-cardinality 
    # to avoid dumping thousands of unique cell barcodes into the LLM context.
    for col in adata.obs.columns:
        n_unique = adata.obs[col].nunique()
        if n_unique <= max_categories:
            # Convert to list for JSON serialization
            unique_vals = adata.obs[col].dropna().unique().tolist()
            context["biological_context"][col] = unique_vals

    # 2. Data Processing State (Raw vs. Normalized)
    # We inspect the actual values in X to guess if it's raw counts
    if adata.X is not None:
        if sp.issparse(adata.X):
            # For sparse matrices, we only look at non-zero elements for speed
            data_arr = adata.X.data
        else:
            data_arr = np.asarray(adata.X)
            
        if len(data_arr) > 0:
            val_min = float(data_arr.min())
            val_max = float(data_arr.max())
            
            # Check if all values are integers (raw counts)
            # We sample up to 10,000 elements for speed
            sample = data_arr[:10000]
            is_integer = bool(np.all(np.equal(np.mod(sample, 1), 0)))
            
            context["data_processing_state"] = {
                "min_value": val_min,
                "max_value": val_max,
                "contains_only_integers": is_integer,
                "likely_raw_counts": is_integer and val_min >= 0
            }
        else:
            context["data_processing_state"] = "Matrix is empty"
            
    # 3. Unstructured Metadata (uns) detailed peek
    # If standard scanpy tools were run, expose their specific parameters
    if 'rank_genes_groups' in adata.uns:
        context["available_structures"]["rank_genes_groups_params"] = adata.uns['rank_genes_groups'].get('params', {})
        
    if 'neighbors' in adata.uns:
        context["available_structures"]["neighbors_params"] = adata.uns['neighbors'].get('params', {})

    return json.dumps(context, indent=2)