

import asyncio
import os
import signal
import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents.graph import  app_graph
from data.ingest_dataset_to_anndata import ingest_dataset_to_anndata
from data.extract_adata_schema import extract_agent_context
import json
from pathlib import Path
from tools.registry import tools
from tools.annotation.AnnotatorSCSA import AnnotatorSCSA
# Initialize FastAPI App
app = FastAPI(
    title="CellAgent API",
    description="Autonomous Single-Cell Analysis Agent",
    version="1.0.0"
)
class AgentRequest(BaseModel):
    instruction: str
    dataset_path: str


def _descendant_pids(root_pid: int) -> list[int]:
    """Return all Linux child-process IDs below ``root_pid``."""
    children: dict[int, list[int]] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            pid = int(status_path.parent.name)
            parent_line = next(
                line for line in status_path.read_text().splitlines()
                if line.startswith("PPid:")
            )
            parent = int(parent_line.split()[1])
        except (OSError, StopIteration, ValueError):
            continue
        children.setdefault(parent, []).append(pid)

    descendants: list[int] = []
    pending = list(children.get(root_pid, []))
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children.get(pid, []))
    return descendants


def _terminate_everything() -> None:
    """Kill agent workers, notebook kernels, subprocesses, then this server."""
    time.sleep(0.25)  # Give the stop endpoint enough time to send its response.
    descendants = _descendant_pids(os.getpid())
    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    time.sleep(0.5)
    remaining = set(descendants) | set(_descendant_pids(os.getpid()))
    for pid in reversed(list(remaining)):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    os.kill(os.getpid(), signal.SIGKILL)


@app.post("/api/stop-all")
async def stop_all_agent_processes():
    """Immediately stop every active agent run and the API server itself."""
    threading.Thread(
        target=_terminate_everything,
        name="cellagent-hard-stop",
        daemon=False,
    ).start()
    return {"status": "stopping", "message": "Stopping all CellAgent processes."}


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

    # Keep FastAPI's event loop responsive so /api/stop-all can be handled
    # while LangGraph or an isolated notebook kernel is still running.
    result = await asyncio.to_thread(app_graph.invoke, initial_state)
    
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
