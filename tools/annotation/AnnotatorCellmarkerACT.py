import pandas as pd
import scanpy as sc
import anndata
import psycopg2 # Required for PostgreSQL connection
import os

def AnnotatorCellmarkerACT(
    adata: anndata.AnnData, 
    species: str, 
    tissue_type: str, 
    db_uri: str | None = None,
    cluster_key: str = "leiden", 
    top_n: int = 50
) -> anndata.AnnData:
    """
    AnnotatorCellmarkerACT(adata, species, tissue_type, db_uri, cluster_key='leiden', top_n=50) -> anndata.AnnData
    
    Annotates single-cell clusters by comparing top Differentially Expressed Genes (DEGs) 
    against the CellMarker 2.0 PostgreSQL database.
    
    Parameters:
    - adata: anndata.AnnData containing the scRNA-seq data.
    - species: str, e.g., 'Human' or 'Mouse'.
    - tissue_type: str, e.g., 'Pancreas' or 'Brain'.
    - db_uri: str, PostgreSQL connection URI (e.g., 'postgresql://user:pass@host:port/dbname').
    - cluster_key: str, the column in adata.obs containing cluster assignments.
    - top_n: int, number of top DEGs to extract per cluster for scoring.
    
    Returns:
    - adata: Updated AnnData object with a new column 'cell_type_act' in adata.obs.
    """
    
    print(f"Querying CellMarker PostgreSQL database for {species} {tissue_type}...")
    
    # 1. Connect and Execute Raw SQL
    # Ensure your actual table name matches the query below (using 'cell_markers' as a placeholder)
    try:
        db_uri = db_uri or os.environ.get("CELLAGENT_MARKER_DB_URI")
        if not db_uri:
            raise ValueError(
                "Provide db_uri or set the CELLAGENT_MARKER_DB_URI environment variable."
            )
        conn = psycopg2.connect(db_uri)
        
        query = """
            SELECT cell_name, marker 
            FROM cell_markers 
            WHERE LOWER(species) = LOWER(%s) 
              AND LOWER(tissue_class) = LOWER(%s);
        """
        
        # pandas read_sql handles the raw query and parameterized inputs securely
        df_filtered = pd.read_sql(query, conn, params=(species, tissue_type))
        
    except Exception as e:
        raise RuntimeError(f"Database connection or SQL execution failed: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            
    if df_filtered.empty:
        raise ValueError(f"No markers found in database for species='{species}' and tissue='{tissue_type}'.")

    # 2. Build Dictionary: { "Cell Type": set(genes) }
    cell_markers = {}
    for _, row in df_filtered.iterrows():
        cell_name = row['cell_name']
        genes = [g.strip() for g in str(row['marker']).split(',')]
        if cell_name not in cell_markers:
            cell_markers[cell_name] = set()
        cell_markers[cell_name].update(genes)

    # 3. Compute DEGs if not already computed
    if f"rank_genes_groups" not in adata.uns or adata.uns["rank_genes_groups"]["params"]["groupby"] != cluster_key:
        print(f"Computing rank_genes_groups for {cluster_key}...")
        sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')

    # 4. Extract Top Genes & Score
    cluster_annotations = {}
    clusters = adata.obs[cluster_key].unique()
    
    for cluster in clusters:
        top_genes = set(pd.DataFrame(adata.uns['rank_genes_groups']['names'])[cluster].head(top_n))
        
        best_score = -1
        best_cell_type = "Unknown"
        
        for cell_type, marker_genes in cell_markers.items():
            overlap = len(top_genes.intersection(marker_genes))
            if overlap > best_score:
                best_score = overlap
                best_cell_type = cell_type
        
        if best_score == 0:
            cluster_annotations[cluster] = "Unknown"
        else:
            cluster_annotations[cluster] = best_cell_type

    # 5. Map annotations back to AnnData
    adata.obs['cell_type_act'] = adata.obs[cluster_key].map(cluster_annotations)
    
    print(f"ACT Annotation Complete. Mapped types: {cluster_annotations}")
    return adata
