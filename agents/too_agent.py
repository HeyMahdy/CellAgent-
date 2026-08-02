

from langchain_openai import ChatOpenAI
from agents.memory import CellAgentState
from pydantic import BaseModel, Field
from typing import Dict, Any
import json
from langchain_core.messages import SystemMessage, HumanMessage

class ToolSelection(BaseModel):
    selected_tool_name: str = Field(
        description="An exact tool name from the registry, or the string NONE if no specialized tool applies."
    )

TOOL_SELECTOR_PROMPT = """You are the Tool Selector agent in a single-cell RNA-seq pipeline.

ROLE:
Analyze the user's current subtask and the provided tool registry. Determine which specialized tool, if any, is needed to execute this subtask.

RULES:
1. Select at most ONE relevant tool that best fits the subtask requirements.
2. For standard preprocessing tasks (e.g., QC, Normalization, PCA, Clustering) that are handled by standard libraries, return null UNLESS a specific tool in the registry is explicitly designed for that task.
3. If you select a tool, output its exact name from the registry.
4. Output the string NONE if no tool is required.
"""

_tool_selector_structured_llm = None


def get_tool_selector_llm():
    """Create the client lazily so importing the graph needs no API key."""
    global _tool_selector_structured_llm
    if _tool_selector_structured_llm is None:
        tool_selector_llm = ChatOpenAI(model="gpt-5.4-mini")
        _tool_selector_structured_llm = tool_selector_llm.with_structured_output(ToolSelection)
    return _tool_selector_structured_llm


def tool_selector_node(state: CellAgentState) -> dict:
    """
    Dynamically selects a single tool from the registry based on the current subtask.
    """
    current_task = state.get("current_task")
    tools_registry = state.get("tools_registry", [])

    # Safety check: if there's no task or no tools, skip
    if not current_task or not tools_registry:
        return {"active_tools": []}

    user_content = (
        f"CURRENT SUBTASK:\n"
        f"Action: {current_task.get('action')}\n"
        f"Rationale: {current_task.get('rationale')}\n\n"
        f"TOOL REGISTRY:\n{json.dumps(tools_registry, indent=2)}"
    )

    messages = [
        SystemMessage(content=TOOL_SELECTOR_PROMPT),
        HumanMessage(content=user_content)
    ]

    selection_result: ToolSelection = get_tool_selector_llm().invoke(messages)
    selected_name = selection_result.selected_tool_name.strip()
    if selected_name.upper() == "NONE":
        return {"tool_info": None}

    selected_tool = next(
        (tool for tool in tools_registry if tool.get("name") == selected_name),
        None,
    )
    if selected_tool is None:
        raise ValueError(f"Tool Selector returned an unregistered tool name: {selected_name!r}")
    return {"tool_info": selected_tool}
