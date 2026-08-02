
from agents.memory import CellAgentState
from agents.planner import planner_node
from agents.loop import loop_node, route_from_loop
from agents.too_agent import tool_selector_node
from agents.code_programmer import code_programmer_node
from agents.sandbox import sandbox_node, route_after_sandbox
from agents.annotation_output import annotation_output_node
from agents.visualization_output import visualization_output_node
from langgraph.graph import END, START, StateGraph

workflow = StateGraph(CellAgentState)

# Add your nodes
workflow.add_node("planner_node", planner_node)
workflow.add_node("loop_node", loop_node)
workflow.add_node("tool_selector_node", tool_selector_node)
workflow.add_node("code_programmer_node", code_programmer_node)
workflow.add_node("sandbox_node", sandbox_node)
workflow.add_node("annotation_output_node", annotation_output_node)
workflow.add_node("visualization_output_node", visualization_output_node)

# 1. Start -> Planner -> Loop
workflow.add_edge(START, "planner_node")
workflow.add_edge("planner_node", "loop_node")

# 2. Loop conditionally goes to Executor or END
workflow.add_conditional_edges(
    "loop_node",
    route_from_loop,
    {
        "tool_selector_node": "tool_selector_node",
        "annotation_output_node": "annotation_output_node",
    },
)

# 3. For each task: select a tool, generate code, then execute it in a fresh
# notebook kernel. A failed execution is retried twice with its error history.
workflow.add_edge("tool_selector_node", "code_programmer_node")
workflow.add_edge("code_programmer_node", "sandbox_node")
workflow.add_conditional_edges(
    "sandbox_node",
    route_after_sandbox,
    {
        "code_programmer_node": "code_programmer_node",
        "loop_node": "loop_node",
    },
)
workflow.add_edge("annotation_output_node", "visualization_output_node")
workflow.add_edge("visualization_output_node", END)

app_graph = workflow.compile()
