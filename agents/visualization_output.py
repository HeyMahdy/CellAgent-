"""Deterministic final visualization stage for annotated AnnData artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


CANONICAL_MARKERS = (
    "CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "MS4A1", "CD79A",
    "CD74", "LYZ", "LST1", "FCN1", "S100A8", "S100A9", "LILRA4",
    "GZMB", "JCHAIN", "MZB1", "HBB",
)


def _expression(adata: ad.AnnData, gene: str) -> np.ndarray:
    values = adata[:, gene].X
    return values.toarray().ravel() if sp.issparse(values) else np.asarray(values).ravel()


def _marker_genes(adata: ad.AnnData, limit: int = 6) -> list[str]:
    lookup = {str(gene).upper(): str(gene) for gene in adata.var_names}
    return [lookup[marker] for marker in CANONICAL_MARKERS if marker in lookup][:limit]


def _embedding(adata: ad.AnnData) -> tuple[np.ndarray, str]:
    if "X_umap" in adata.obsm:
        return np.asarray(adata.obsm["X_umap"]), "existing X_umap"
    if "X_pca" not in adata.obsm:
        raise ValueError("The annotated artifact has neither X_umap nor X_pca for visualization.")
    # Keep visualization reliable even when a previous generated step did not
    # persist UMAP coordinates. The response explicitly identifies this as a
    # PCA fallback rather than misrepresenting it as a UMAP embedding.
    return np.asarray(adata.obsm["X_pca"])[:, :2], "PCA projection fallback (X_umap unavailable)"


def _plot_annotated_umap(
    embedding: np.ndarray, adata: ad.AnnData, label_key: str, output: Path, title: str
) -> None:
    labels = adata.obs[label_key].astype(str)
    categories = sorted(labels.unique())
    palette = plt.get_cmap("tab20", len(categories))
    fig, ax = plt.subplots(figsize=(10, 7))
    for index, category in enumerate(categories):
        mask = labels.eq(category).to_numpy()
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.7,
                   color=palette(index), label=category, linewidths=0)
    ax.set(title=title, xlabel="Embedding 1", ylabel="Embedding 2")
    ax.legend(title="Cell type", bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=3)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_umaps(embedding: np.ndarray, adata: ad.AnnData, genes: list[str], output: Path) -> None:
    columns = min(3, len(genes))
    rows = int(np.ceil(len(genes) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for ax, gene in zip(axes.ravel(), genes):
        expression = _expression(adata, gene)
        plot = ax.scatter(embedding[:, 0], embedding[:, 1], c=expression, s=4,
                          cmap="viridis", linewidths=0)
        ax.set(title=gene, xlabel="UMAP 1", ylabel="UMAP 2")
        fig.colorbar(plot, ax=ax, label="Expression")
    for ax in axes.ravel()[len(genes):]:
        ax.axis("off")
    fig.suptitle("Marker-gene expression on UMAP", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_violins(adata: ad.AnnData, label_key: str, genes: list[str], output: Path) -> None:
    labels = adata.obs[label_key].astype(str)
    categories = sorted(labels.unique())
    fig, axes = plt.subplots(len(genes), 1, figsize=(max(10, len(categories) * 1.2), 3.8 * len(genes)), squeeze=False)
    for ax, gene in zip(axes.ravel(), genes):
        expression = _expression(adata, gene)
        values = [expression[labels.eq(category).to_numpy()] for category in categories]
        ax.violinplot(values, showmedians=True, showextrema=False)
        ax.set(title=f"{gene} expression by consensus cell type", ylabel="Expression")
        ax.set_xticks(range(1, len(categories) + 1), categories, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def visualization_output_node(state: dict[str, Any]) -> dict[str, Any]:
    """Create PNGs from the final annotated AnnData artifact, when available."""
    annotation = state.get("final_annotation_output") or {}
    artifact_path = annotation.get("annotated_adata_path")
    if annotation.get("status") != "ok" or not artifact_path:
        return {"visualization_output": {"status": "skipped", "message": "No successful annotation artifact is available."}}

    artifact = Path(artifact_path)
    try:
        adata = ad.read_h5ad(artifact)
        candidates = annotation.get("annotation_columns", [])
        label_key = next((key for key in candidates if key in adata.obs.columns and not key.endswith("_source")), None)
        if label_key is None:
            raise ValueError("No final cell-type label column was found in the annotated artifact.")
        embedding, embedding_source = _embedding(adata)

        output_dir = artifact.parent
        umap_path = output_dir / "final_annotated_umap.png"
        title = "Annotated UMAP" if embedding_source == "existing X_umap" else "Annotated PCA projection"
        _plot_annotated_umap(embedding, adata, label_key, umap_path, title)

        paths = [str(umap_path)]
        markers = _marker_genes(adata)
        if markers:
            feature_path = output_dir / "marker_feature_umaps.png"
            violin_path = output_dir / "marker_expression_violins.png"
            _plot_feature_umaps(embedding, adata, markers, feature_path)
            _plot_violins(adata, label_key, markers[:4], violin_path)
            paths.extend([str(feature_path), str(violin_path)])

        return {"visualization_output": {"status": "ok", "label_key": label_key, "embedding_source": embedding_source, "marker_genes": markers, "png_paths": paths}}
    except Exception as exc:
        return {"visualization_output": {"status": "error", "message": f"{type(exc).__name__}: {exc}"}}
