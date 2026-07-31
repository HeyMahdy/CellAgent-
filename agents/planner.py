
from typing import TypedDict, Optional, Literal
import json
from langchain_core.messages import SystemMessage, HumanMessage
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from agents.memory import CellAgentState

PLANNER_SYSTEM_PROMPT = """You are the Planner in a single-cell RNA-seq analysis pipeline.
 
ROLE:
Your only job is to decompose the user's task into an ordered sequence of
subtasks. You do NOT execute anything, call any tool, or touch the dataset
directly. A separate executor component will read your subtask list and
carry out each step. You are given a description of the dataset (not the
raw data) — reason from that description only.
 
OUTPUT FORMAT (mandatory):
Respond with ONLY a JSON array of subtask objects, nothing else — no prose
before or after. Each subtask object has this shape:
[
  {"step": 1, "action": "<short imperative description>", "rationale": "<why this step, in one sentence>"},
  {"step": 2, "action": "...", "rationale": "..."}
]
Do not wrap the array in markdown code fences. Do not add commentary outside
the array.
 
EXPERT EXPERIENCE (standard single-cell processing order):
A typical scRNA-seq pipeline follows this general order, though not every
task needs every stage:
1. Quality control — filter low-quality cells/genes (e.g. low counts,
   high mitochondrial fraction) before anything else.
2. Normalization — correct for sequencing depth differences between cells;
   required before most downstream comparisons or annotation. Check the
   dataset's data_processing_state first — skip this step if the data is
   already normalized.
3. Highly variable gene (HVG) selection — reduces noise before
   dimensionality reduction; usually precedes PCA/clustering.
4. Batch correction — only needed if the dataset spans multiple batches,
   samples, or patients (check obs_columns/biological_context for a
   batch-like column) and downstream comparisons would otherwise be
   confounded by batch effects.
5. Dimensionality reduction & clustering — PCA, neighbor graph, Leiden/
   Louvain clustering; required before most annotation methods, which
   operate per-cluster.
6. Cell-type annotation — requires clusters to exist first (see step 5) and
   requires normalized (not raw) data.
7. Trajectory inference / further analysis — only if the user's task asks
   about developmental progression, pseudotime, or similar; requires
   clustering and often annotation to already exist.
 
Only include the subtasks actually needed for the user's specific request —
do not pad the plan with irrelevant standard steps. If the dataset profile
shows a step has already been done (e.g. clustering already exists in
obs_columns, or data is already normalized), skip it and say so in the
rationale rather than repeating it.


"""


class Subtask(BaseModel):
    step: int = Field(description="1-indexed order of this subtask")
    action: str = Field(description="Short imperative description of the step")
    rationale: str = Field(description="One sentence on why this step is needed here")
    tool_name: Optional[str] = Field(description="The exact name of the tool from the tools_registry to use for this step, if applicable. Otherwise, null.", default=None)
 
 
class Plan(BaseModel):
    subtasks: list[Subtask]



llm = ChatOpenAI(model="gpt-5.4-mini", api_key="")

structured_llm = llm.with_structured_output(Plan)


def planner_node(state: CellAgentState) -> dict:
    """
    Analyzes the user query and dataset metadata to generate a step-by-step plan.
    """
    query = state["query"]
    metadata = state["dataset_metadata"]
    tools = state["tools_registry"]
    
    # Construct the user message with exact details for the LLM to reason over
    user_content = (
        f"USER TASK:\n{query}\n\n"
        f"DATASET PROFILE:\n{json.dumps(metadata, indent=2)}\n\n"
        f"AVAILABLE TOOLS:\n{json.dumps(tools, indent=2)}\n\n"
        "If a subtask aligns with a tool provided in the AVAILABLE TOOLS list, "
        "you MUST include the exact tool name in the 'tool_name' field."
    )
    
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]
    
    # Invoke the LLM - structured_llm guarantees it returns a Plan object
    plan_result: Plan = structured_llm.invoke(messages)
    
    # Convert the Pydantic objects to standard dictionaries for the LangGraph state
    # This prevents serialization issues if you ever add a checkpointer to the graph
    subtasks = [subtask.model_dump() for subtask in plan_result.subtasks]
    
    return {"planner_output": subtasks}
    