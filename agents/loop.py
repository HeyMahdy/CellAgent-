

import json
from langchain_core.messages import SystemMessage, HumanMessage
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from agents.memory import CellAgentState


def loop_node(state: CellAgentState) -> dict:
    """
    Grabs the next task from the planner_output list.
    """
    task_list = state.get("planner_output", [])
    current_index = state.get("current_task_index", 0)

    # Check if there are still tasks left to execute
    if current_index < len(task_list):
        next_task = task_list[current_index]

        # Return the task for the executor to read, and increment the index
        return {
            "current_task": next_task,
            "current_task_index": current_index + 1
        }
    else:
        # Loop is finished, no tasks remain
        return {
            "current_task": None
        }


def route_from_loop(state: CellAgentState) -> str:
    """
    Decides whether to continue the loop or end the graph.
    """
    if state.get("current_task") is None:
        return "annotation_output_node"
    else:
        return "tool_selector_node"
