


tools = [
  {
    "name": "AnnotatorCellmarkerACT",
    "description": "Labels each cluster with a cell type by taking its top differentially expressed genes and counting how many overlap with known marker gene sets, pulled live from a CellMarker 2.0 PostgreSQL database. Use when a marker-based approach is preferred and a CellMarker database connection is available — this is a simpler overlap-count method than SCSA's statistical scoring.",
    "import_statement": "import os\nimport pandas as pd\nimport scanpy as sc\nimport anndata\nimport psycopg2\nfrom tools.annotation.AnnotatorCellmarkerACT import AnnotatorCellmarkerACT",
    "parameters": {
      "species": "string — e.g. 'Human' or 'Mouse'. Must exactly match a species value in the database (case-insensitive); no fuzzy matching.",
      "tissue_type": "string — e.g. 'Pancreas' or 'Brain'. Must exactly match a tissue_class value in the database (case-insensitive); no fuzzy matching.",
      "db_uri": "string or null — PostgreSQL connection URI. If omitted, falls back to the CELLAGENT_MARKER_DB_URI environment variable; fails immediately if neither is set.",
      "cluster_key": "string — obs column with cluster assignments (e.g. 'leiden'). Required — annotation is per-cluster.",
      "top_n": "int — number of top DEGs per cluster to check for marker overlap. Higher values are more lenient but noisier."
    },
    "returns": "the same AnnData object, with a new column 'cell_type_act' added to obs (per-cell, but shared across a cluster; clusters with zero marker overlap get 'Unknown')",
    "preconditions": "cluster_key must already exist in obs_columns. Does NOT require pre-computed rank_genes_groups — computes it automatically if missing or computed against a different cluster_key. Requires network access to a live PostgreSQL database — either db_uri passed explicitly or CELLAGENT_MARKER_DB_URI set as an environment variable.",
    "usage_example": "import os\nimport pandas as pd\nimport scanpy as sc\nimport anndata\nimport psycopg2\nfrom tools.annotation.AnnotatorCellmarkerACT import AnnotatorCellmarkerACT\n\nif 'leiden' not in adata.obs.columns:\n    raise ValueError(\"Cluster key 'leiden' missing.\")\nadata = AnnotatorCellmarkerACT(adata=adata, species='Human', tissue_type='All', cluster_key='leiden', top_n=10)"
  },
  {
    "name": "AnnotatorCellTypist",
    "description": "Automatically labels each cell with a cell type (e.g. T-cell, B-cell, monocyte) using CellTypist's pre-trained models. Fast, no external dependencies. Use when the user wants cell type labels and doesn't need custom/tissue-specific reasoning.",
    "import_statement": "import anndata\nimport celltypist\nfrom celltypist import models\nfrom tools.annotation.AnnotatorCellTypist import AnnotatorCellTypist",
    "parameters": {
      "model_name": "string — pre-trained CellTypist model name, e.g. 'Immune_All_Low.pkl'. Defaults to broad immune reference; pick a more specific model if tissue/context is known.",
      "majority_voting": "boolean — if true, smooths predictions using cluster consensus instead of per-cell labels. Requires cluster_key to already exist in obs.",
      "cluster_key": "string — obs column with cluster assignments (e.g. 'leiden'). Only used if majority_voting=true."
    },
    "returns": "the same AnnData object, with a new column 'cell_type_celltypist' added to obs",
    "preconditions": "Data must be log1p-normalized. If majority_voting=true, cluster_key must exist in obs_columns.",
    "usage_example": "import anndata\nimport celltypist\nfrom celltypist import models\nfrom tools.annotation.AnnotatorCellTypist import AnnotatorCellTypist\n\nif 'log1p' not in adata.uns:\n    raise ValueError(\"Data must be log1p normalized.\")\nadata = AnnotatorCellTypist(adata=adata, model_name='Immune_All_Low.pkl', majority_voting=True, cluster_key='leiden')"
  },
  {
    "name": "AnnotatorSCSA",
    "description": "Labels each cluster (not each cell) with a cell type by computing differential expression genes per cluster and matching them against a marker-gene database via the external SCSA tool. Slower and requires local setup, but useful when CellTypist's pre-trained models don't cover the tissue/species in question, or when marker-gene-based reasoning is preferred over a pre-trained classifier.",
    "import_statement": "import os\nimport tempfile\nimport subprocess\nimport pandas as pd\nimport scanpy as sc\nimport anndata\nfrom tools.annotation.AnnotatorSCSA import AnnotatorSCSA",
    "parameters": {
      "scsa_script_path": "string — local filesystem path to SCSA.py. Must already exist on disk; this tool cannot install or locate it.",
      "scsa_db_path": "string — local filesystem path to the SCSA marker database file (e.g. whole_v2.db). Must already exist on disk.",
      "species": "string — 'Human' or 'Mouse'. Affects which markers in the database are matched.",
      "tissue": "string — restricts marker matching to a specific tissue; 'All' searches every tissue in the database.",
      "cluster_key": "string — obs column with cluster assignments (e.g. 'leiden'). Required — annotation is per-cluster, not per-cell.",
      "foldchange_thresh": "float — minimum log2 fold-change for a gene to count as a cluster marker.",
      "pval_thresh": "float — maximum p-value for a gene to count as a cluster marker."
    },
    "returns": "the same AnnData object, with a new column 'cell_type_scsa' added to obs (per-cell, but all cells in the same cluster share the same label; unmatched clusters get 'Unknown')",
    "preconditions": "cluster_key must already exist in obs_columns (clustering must have been run first). Does NOT require pre-computed rank_genes_groups. Requires scsa_script_path and scsa_db_path to point to files that actually exist on the local machine.",
    "usage_example": "import os\nimport tempfile\nimport subprocess\nimport pandas as pd\nimport scanpy as sc\nimport anndata\nfrom tools.annotation.AnnotatorSCSA import AnnotatorSCSA\n\nscsa_script = os.environ.get('SCSA_SCRIPT_PATH')\nscsa_db = os.environ.get('SCSA_DB_PATH')\nif not scsa_script or not scsa_db:\n    raise ValueError(\"SCSA environment variables missing.\")\nif 'leiden' not in adata.obs.columns:\n    raise ValueError(\"Cluster key 'leiden' missing.\")\nadata = AnnotatorSCSA(adata=adata, scsa_script_path=scsa_script, scsa_db_path=scsa_db, species='Human', tissue='All', cluster_key='leiden', foldchange_thresh=1.0, pval_thresh=0.05)"
  }
]