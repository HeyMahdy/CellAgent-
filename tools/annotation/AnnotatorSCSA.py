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
    """
    print("Preparing DEG data for SCSA...")

    # 1. Ensure DEGs are calculated
    if "rank_genes_groups" not in adata.uns or adata.uns["rank_genes_groups"]["params"]["groupby"] != cluster_key:
        print(f"Computing rank_genes_groups for {cluster_key}...")
        sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')

    # 2. Extract DEGs into a Scanpy-compatible DataFrame (Wide format for SCSA)
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    scsa_data = {}
    for group in groups:
        scsa_data[f"{group}_names"] = result['names'][group]
        scsa_data[f"{group}_logfoldchanges"] = result['logfoldchanges'][group]
        scsa_data[f"{group}_pvals"] = result['pvals'][group]
        scsa_data[f"{group}_pvals_adj"] = result['pvals_adj'][group]
        if 'scores' in result:
            scsa_data[f"{group}_scores"] = result['scores'][group]
            
    deg_df = pd.DataFrame(scsa_data)
    
    # 3. Create temporary directory to hold I/O files
    with tempfile.TemporaryDirectory() as temp_dir:
        input_csv = os.path.join(temp_dir, "deg_input.csv")
        output_txt = os.path.join(temp_dir, "scsa_out.txt") # <-- FIXED: Add .txt directly here
        
        # IMPORTANT: SCSA uses index_col=0, so index=True is required
        deg_df.to_csv(input_csv, index=True)
        
        # 4. Construct SCSA Command Line Execution
        cmd = [
            "python", scsa_script_path,
            "-d", scsa_db_path,
            "-i", input_csv,
            "-s", "scanpy",
            "-o", output_txt, # <-- Pass the exact .txt path to SCSA
            "-m", "txt",
            "-f", str(foldchange_thresh),
            "-p", str(pval_thresh),
            "-g", species,
            "-E"  # <-- FIXED: Tells SCSA to use Gene Symbols instead of Ensembl IDs
        ]
        
        # Add tissue filter if provided
        if tissue.lower() != "all":
            cmd.extend(["-k", tissue])
            
        print(f"Running SCSA command: {' '.join(cmd)}")
        
        # Run SCSA and capture ALL output for diagnostics
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            raise RuntimeError(f"SCSA crashed.\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")

        # 5. Parse the SCSA Output
        if not os.path.exists(output_txt):
            found_files = os.listdir(temp_dir)
            raise FileNotFoundError(
                f"SCSA finished, but no output file '{output_txt}' was found.\n\n"
                f"--- SCSA STDOUT ---\n{process.stdout}\n\n"
                f"--- SCSA STDERR ---\n{process.stderr}\n\n"
                f"--- FILES IN TEMP FOLDER ---\n{found_files}"
            )
            
        # SCSA output typically contains 'Cluster', 'Cell Type', 'Score', etc.
        scsa_res = pd.read_csv(output_txt, sep='\t')
        
        cluster_annotations = {}
        if not scsa_res.empty:
            for cluster, group in scsa_res.groupby('Cluster'):
                top_cell_type = group.iloc[0]['Cell Type']
                cluster_annotations[str(cluster)] = top_cell_type
        else:
            print("Warning: SCSA returned an empty file (thresholds might be too strict).")

    # 6. Map annotations back to AnnData
    adata.obs['cell_type_scsa'] = adata.obs[cluster_key].astype(str).map(cluster_annotations)
    adata.obs['cell_type_scsa'] = adata.obs['cell_type_scsa'].fillna("Unknown")
    
    print(f"SCSA Annotation Complete. Mapped types: {cluster_annotations}")
    return adata