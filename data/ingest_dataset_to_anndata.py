import os
import anndata as ad
import scanpy as sc

def ingest_dataset_to_anndata(path: str) -> ad.AnnData:
    """
    Detects the format of a single-cell dataset from a path and loads it into a standardized AnnData object.
    
    Args:
        path (str): The file path or directory containing the dataset.
        
    Returns:
        AnnData: The standardized dataset object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")

    # 1. Handle 10x Genomics Directories
    if os.path.isdir(path):
        # Scanpy automatically looks for matrix.mtx, barcodes.tsv, and features/genes.tsv
        print(f"Directory detected. Attempting to load as 10x Genomics matrix from {path}...")
        try:
            # var_names='gene_symbols' is standard, but some older datasets use 'gene_ids'
            adata = sc.read_10x_mtx(path, var_names='gene_symbols', cache=True)
            return adata
        except Exception as e:
            raise ValueError(f"Failed to parse directory as a 10x dataset. Error: {e}")

    # 2. Handle specific file extensions
    _, ext = os.path.splitext(path.lower())

    if ext == '.h5ad':
        print("Native AnnData file detected. Loading...")
        return ad.read_h5ad(path)

    elif ext == '.h5':
        # 10x Genomics also distributes data as compressed HDF5 files (.h5)
        print("HDF5 file detected. Attempting to load as 10x Genomics H5...")
        return sc.read_10x_h5(path)

    elif ext == '.csv':
        print("CSV file detected. Loading...")
        # Note: Depending on how the biologist saved it, you might need to transpose the matrix.
        # If genes are rows and cells are columns, you would call adata = adata.T after loading.
        return ad.read_csv(path)

    elif ext == '.mtx':
        print("Standalone Matrix Market file detected. Loading...")
        # This only loads the numbers. Metadata (cells/genes) will be missing unless loaded separately.
        return ad.read_mtx(path)
        
    elif ext in ['.rds', '.robj']:
        raise NotImplementedError(
            "R objects (.rds) cannot be loaded directly in Python. "
            "Please convert them to .h5ad in R using the 'reticulate' or 'SeuratDisk' packages first."
        )

    else:
        raise ValueError(f"Unsupported file format: {ext}")