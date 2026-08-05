"""LLM-based biological consensus evaluator."""

import json

import anndata as ad
import pandas as pd
import scanpy as sc
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.memory import CellAgentState
from agents.evaluator_prompt import EVALUATOR_SYSTEM_PROMPT


system_prompt = EVALUATOR_SYSTEM_PROMPT


class ClusterConsensus(BaseModel):
    cluster_id: str = Field(description="Exact cluster identifier from the input.")
    final_label: str
    supporting_markers: list[str] = Field(default_factory=list)
    justification: str


class EvaluationResult(BaseModel):
    clusters: list[ClusterConsensus]


llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
structured_llm = llm.with_structured_output(EvaluationResult)

ANNOTATED_ADATA_PATH = "/content/cellagent_artifacts/step_5_adata.h5ad"


def _json_value(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _build_evaluator_input(state: CellAgentState) -> dict:
    adata = ad.read_h5ad(ANNOTATED_ADATA_PATH)
    if "rank_genes_groups" not in adata.uns:
        raise ValueError("The final AnnData artifact is missing rank_genes_groups.")

    rank_params = adata.uns["rank_genes_groups"].get("params", {})
    cluster_key = rank_params.get("groupby")
    if not cluster_key or cluster_key not in adata.obs.columns:
        raise ValueError("rank_genes_groups does not reference a valid cluster column.")

    annotation_columns = {
        "celltypist": "cell_type_celltypist",
        "act": "cell_type_act",
        "scsa": "cell_type_scsa",
    }
    missing = [column for column in annotation_columns.values() if column not in adata.obs.columns]
    if missing:
        raise ValueError(f"The final AnnData artifact is missing annotation columns: {missing}")

    cluster_values = adata.obs[cluster_key].astype(str)
    clusters = []
    for cluster_id in cluster_values.unique():
        mask = cluster_values == cluster_id
        n_cells = int(mask.sum())

        marker_frame = sc.get.rank_genes_groups_df(adata, group=cluster_id).head(20)
        top_markers = []
        for rank, (_, row) in enumerate(marker_frame.iterrows(), start=1):
            marker = {"gene": str(row["names"]), "rank": rank}
            marker_fields = {
                "scores": "score",
                "logfoldchanges": "logfoldchange",
                "pvals": "pval",
                "pvals_adj": "pval_adj",
            }
            for source_field, output_field in marker_fields.items():
                if source_field in marker_frame.columns:
                    marker[output_field] = _json_value(row[source_field])
            top_markers.append(marker)

        tool_predictions = {}
        for source, column in annotation_columns.items():
            counts = adata.obs.loc[mask, column].astype(str).value_counts(dropna=False)
            tool_predictions[source] = [
                {
                    "label": str(label),
                    "count": int(count),
                    "fraction": round(int(count) / n_cells, 6),
                }
                for label, count in counts.items()
            ]

        clusters.append({
            "cluster_id": str(cluster_id),
            "n_cells": n_cells,
            "top_markers": top_markers,
            "tool_predictions": tool_predictions,
        })

    return {
        "cluster_key": str(cluster_key),
        "marker_groupby": str(cluster_key),
        "clusters": clusters,
    }


def evaluator_node(state: CellAgentState) -> dict:
    """Build annotation evidence and return cluster consensus labels."""
    evaluator_input = _build_evaluator_input(state)

    user_content = (
        f"USER REQUEST:\n{state['query']}\n\n"
        f"DATASET CONTEXT:\n{json.dumps(state['dataset_metadata'], indent=2, default=str)}\n\n"
        f"EVALUATION EVIDENCE:\n{json.dumps(evaluator_input, indent=2, default=str)}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    result: EvaluationResult = structured_llm.invoke(messages)
    return {
        "evaluator_input": evaluator_input,
        "evaluator_output": result.model_dump(),
    }
