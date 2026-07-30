

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import APIRouter, CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from data.ingest_dataset_to_anndata import ingest_dataset_to_anndata

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
    





