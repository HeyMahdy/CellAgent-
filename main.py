

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents.graph import  app_graph
from data.ingest_dataset_to_anndata import ingest_dataset_to_anndata
from data.extract_adata_schema import extract_agent_context
import json
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

    path = payload.dataset_path
    query = payload.instruction
    adata = ingest_dataset_to_anndata(path)
    dataset_context = extract_agent_context(adata)

    initial_state = {
        "query": query,
        "dataset_path": path,
        "adata": adata,
        "dataset_metadata": dataset_context,
        "tools_registry": tools,
        "planner_output": [] # Empty initially, to be filled by the planner node
    }

    result = app_graph.invoke(initial_state)
    
    # Print the resulting plan
    print("=== PLANNER OUTPUT ===")
    print(json.dumps(result["planner_output"], indent=2))

    






