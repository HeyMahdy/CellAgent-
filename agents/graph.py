
from agents.memory import CellAgentState
from agents.planner import planner_node

import json
from pprint import pprint
from langgraph.graph import StateGraph, START, END

# --- 1. Build the Graph ---
workflow = StateGraph(CellAgentState)

# Add our single node
workflow.add_node("planner", planner_node)

# Define the flow: START -> planner -> END
workflow.add_edge(START, "planner")
workflow.add_edge("planner", END)

# Compile the graph
app_graph = workflow.compile()