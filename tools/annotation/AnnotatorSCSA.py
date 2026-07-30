import anndata
import scanpy as sc
import pandas as pd
import subprocess
import os
import tempfile


def AnnotatorSCSA(
    adata: anndata.AnnData, 
    scsa_script_path: str, 
    scsa_db_path: str,
    species: str = "Human",
    tissue: str = "All",
    cluster_key: str = "leiden",
    foldchange_thresh: float = 1.0,
    pval_thresh: float = 0.05
) -> anndata.AnnData:
    """
    Annotates clusters by extracting Scanpy DEGs and running the external SCSA tool.
    
    Parameters:
    - adata: anndata.AnnData containing the scRNA-seq data.
    - scsa_script_path: str, path to the 'SCSA.py' script.
    - scsa_db_path: str, path to the SCSA SQLite database (e.g., 'whole_v2.db').
    - species: str, 'Human' or 'Mouse'.
    - tissue: str, specific tissue to filter the database.
    - cluster_key: str, the column in adata.obs containing cluster assignments.
    - foldchange_thresh: float, log2 fold-change threshold for DEGs.
    - pval_thresh: float, p-value threshold for DEGs.
    
    Returns:
    - adata: Updated AnnData object with a new column 'cell_type_scsa'.
    """
    print("Preparing DEG data for SCSA...")

    # 1. Ensure DEGs are calculated
    if "rank_genes_groups" not in adata.uns or adata.uns["rank_genes_groups"]["params"]["groupby"] != cluster_key:
        print(f"Computing rank_genes_groups for {cluster_key}...")
        sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')

    # 2. Extract DEGs into a Scanpy-compatible DataFrame
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    df_list = []
    for group in groups:
        df_group = pd.DataFrame({
            'cluster': group,
            'name': result['names'][group],
            'logfc': result['logfoldchanges'][group],
            'pval': result['pvals'][group],
            'pval_adj': result['pvals_adj'][group]
        })
        # Filter based on thresholds
        df_group = df_group[(df_group['logfc'] >= foldchange_thresh) & (df_group['pval'] <= pval_thresh)]
        df_list.append(df_group)
        
    deg_df = pd.concat(df_list, ignore_index=True)
    
    # 3. Create temporary directory to hold I/O files
    with tempfile.TemporaryDirectory() as temp_dir:
        input_csv = os.path.join(temp_dir, "deg_input.csv")
        output_prefix = os.path.join(temp_dir, "scsa_out")
        
        deg_df.to_csv(input_csv, index=False)
        
        # 4. Construct SCSA Command Line Execution
        cmd = [
            "python", scsa_script_path,
            "-d", scsa_db_path,
            "-i", input_csv,
            "-s", "scanpy", # Tells SCSA to parse scanpy header format
            "-o", output_prefix,
            "-m", "txt", # Output as text table
            "-b", "foldchange",
            "-p", "pval_adj"
        ]
        
        # Add tissue/species filters if provided
        if tissue.lower() != "all":
            cmd.extend(["-t", tissue])
        if species.lower() == "mouse":
            cmd.extend(["-E", "Mouse"]) # SCSA uses -E for species sometimes depending on the version
            
        print(f"Running SCSA command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SCSA execution failed.\nStdout: {e.stdout}\nStderr: {e.stderr}")

        # 5. Parse the SCSA Output
        output_txt = f"{output_prefix}.txt"
        if not os.path.exists(output_txt):
            raise FileNotFoundError(f"SCSA output file {output_txt} not found. SCSA may have failed silently.")
            
        # SCSA output typically contains 'Cluster', 'Cell Type', 'Score', etc.
        scsa_res = pd.read_csv(output_txt, sep='\t')
        
        # Map highest scoring cell type per cluster
        # SCSA usually sorts by score descending, so taking the first match per cluster works
        cluster_annotations = {}
        for cluster, group in scsa_res.groupby('Cluster'):
            top_cell_type = group.iloc[0]['Cell Type']
            cluster_annotations[str(cluster)] = top_cell_type

    # 6. Map annotations back to AnnData
    adata.obs['cell_type_scsa'] = adata.obs[cluster_key].astype(str).map(cluster_annotations)
    
    # Fill unmapped clusters with "Unknown"
    adata.obs['cell_type_scsa'] = adata.obs['cell_type_scsa'].fillna("Unknown")
    
    print(f"SCSA Annotation Complete. Mapped types: {cluster_annotations}")
    return adata
