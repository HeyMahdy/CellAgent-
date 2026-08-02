

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents.graph import  app_graph
from data.ingest_dataset_to_anndata import ingest_dataset_to_anndata
from data.extract_adata_schema import extract_agent_context
import json
from pathlib import Path
from tools.registry import tools
# Initialize FastAPI App
app = FastAPI(
    title="CellAgent API",
    description="Autonomous Single-Cell Analysis Agent",
    version="1.0.0"
)
class AgentRequest(BaseModel):
    instruction: str
    dataset_path: str


@app.post("/api/run-agent")
async def run_cell_agent(payload: AgentRequest):
    """
    Trigger the LangGraph autonomous agent workflow.
    """

    # The sandbox executes notebooks from a temporary working directory, so a
    # relative request path would work here but fail inside every sandbox run.
    path = str(Path(payload.dataset_path).expanduser().resolve())
    query = payload.instruction
    adata = ingest_dataset_to_anndata(path)
    dataset_context = extract_agent_context(adata)

    initial_state = {
        "query": query,
        "dataset_path": path,
        "adata": adata,
        "dataset_metadata": dataset_context,
        "tools_registry": tools,
        "planner_output": [], # Empty initially, to be filled by the planner node
        "current_task_index": 0,
        "current_task": None,
        "tool_info": None,
        "code_proposal": None,
        "global_code_memory": [],
        "local_memory": [],
        # Each isolated notebook starts by loading the same source dataset.
        "sandbox_bootstrap_code": (
            "from data.ingest_dataset_to_anndata import ingest_dataset_to_anndata\n"
            f"adata = ingest_dataset_to_anndata({path!r})"
        ),
        "artifact_dir": None,
        "sandbox_timeout_seconds": 900,
        "project_root": str(Path(__file__).resolve().parent),
        "final_annotation_output": None,
        "visualization_output": None,
    }

    result = app_graph.invoke(initial_state)
    
    # Print the resulting plan
    print("=== PLANNER OUTPUT ===")
    print(json.dumps(result["planner_output"], indent=2))

    return {
        "planner_output": result.get("planner_output", []),
        "sandbox_result": result.get("sandbox_result"),
        "artifacts": (result.get("sandbox_result") or {}).get("artifacts", []),
        "failed_attempts": result.get("local_memory", []),
        "final_annotation_output": result.get("final_annotation_output"),
        "visualization_output": result.get("visualization_output"),
    }
