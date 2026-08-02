"""Temporary final-output agent for validating generated cell annotations.

This node is intentionally deterministic (not an LLM evaluator): it inspects
the latest AnnData artifact written by the Code Programmer's sandbox cell and
returns a compact preview suitable for the API response.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad


ANNOTATION_NAME_HINTS = ("cell_type", "annotation", "label", "identity")


def _latest_adata_artifact(state: dict[str, Any]) -> Path | None:
    artifacts: list[str] = []
    for memory_item in state.get("global_code_memory", []):
        artifacts.extend(memory_item.get("artifacts", []))
    artifacts.extend((state.get("sandbox_result") or {}).get("artifacts", []))

    candidates = [Path(path) for path in artifacts if str(path).endswith(".h5ad")]
    candidates = [path for path in candidates if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def annotation_output_node(state: dict[str, Any]) -> dict[str, Any]:
    """Produce the temporary API-ready annotation preview from the final artifact."""
    artifact = _latest_adata_artifact(state)
    if artifact is None:
        return {
            "final_annotation_output": {
                "status": "no_annotation_artifact",
                "message": "No .h5ad artifact was written by a successful sandbox step.",
            }
        }

    try:
        adata = ad.read_h5ad(artifact, backed="r")
        annotation_columns = [
            column
            for column in adata.obs.columns
            if any(hint in column.lower() for hint in ANNOTATION_NAME_HINTS)
        ]
        label_counts = {
            column: {str(label): int(count) for label, count in adata.obs[column].value_counts(dropna=False).items()}
            for column in annotation_columns
        }
        output = {
            "status": "ok" if annotation_columns else "annotation_column_not_found",
            "annotated_adata_path": str(artifact),
            "n_cells": int(adata.n_obs),
            "annotation_columns": annotation_columns,
            "label_counts": label_counts,
        }
        adata.file.close()
        return {"final_annotation_output": output}
    except Exception as exc:
        return {
            "final_annotation_output": {
                "status": "artifact_read_error",
                "annotated_adata_path": str(artifact),
                "message": f"{type(exc).__name__}: {exc}",
            }
        }
