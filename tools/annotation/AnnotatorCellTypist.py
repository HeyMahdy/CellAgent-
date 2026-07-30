import anndata
import scanpy as sc
import celltypist
from celltypist import models


def AnnotatorCellTypist(
    adata: anndata.AnnData, 
    model_name: str = "Immune_All_Low.pkl", 
    majority_voting: bool = True,
    cluster_key: str = "leiden"
) -> anndata.AnnData:
    """
    AnnotatorCellTypist(adata, model_name='Immune_All_Low.pkl', majority_voting=True, cluster_key='leiden') -> anndata.AnnData
    
    Annotates single-cell datasets using CellTypist's pre-trained logistic regression models.
    
    Parameters:
    - adata: anndata.AnnData containing the scRNA-seq data. MUST be log1p normalized to 10k CPM.
    - model_name: str, name of the pre-trained CellTypist model to use.
    - majority_voting: bool, whether to refine predictions based on cluster consensus.
    - cluster_key: str, the column in adata.obs containing cluster assignments (used if majority_voting=True).
    
    Returns:
    - adata: Updated AnnData object with a new column 'cell_type_celltypist' in adata.obs.
    """
    print(f"Running CellTypist with model: {model_name}...")
    
    # 1. Download/Ensure model exists locally
    # CellTypist stores these in ~/.celltypist/data/models
    models.download_models(force_update=False, model=[model_name])
    
    # 2. Run the Annotation
    # Note: CellTypist expects log1p of 10,000 CPM data. 
    # If the user passed raw counts, CellTypist will try to normalize it, but it's best if already done.
    if majority_voting and cluster_key not in adata.obs.columns:
        print(f"Warning: cluster_key '{cluster_key}' not found. Disabling majority voting.")
        majority_voting = False

    # Perform prediction
    predictions = celltypist.annotate(
        adata, 
        model=model_name, 
        majority_voting=majority_voting,
        over_clustering=cluster_key if majority_voting else None
    )
    
    # 3. Extract and map back to AnnData
    # Convert predictions back to an AnnData format to extract the labels
    adata_preds = predictions.to_adata()
    
    if majority_voting:
        adata.obs['cell_type_celltypist'] = adata_preds.obs['majority_voting']
    else:
        adata.obs['cell_type_celltypist'] = adata_preds.obs['predicted_labels']
        
    print("CellTypist Annotation Complete.")
    return adata
