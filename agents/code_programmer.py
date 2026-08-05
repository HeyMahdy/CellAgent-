"""The Code Programmer agent for one CellAgent subtask.

This module deliberately generates code but does not execute it.  Execution is
handled by :mod:`agents.sandbox`, which makes it possible to retry a failed
attempt with a concise error record instead of giving the model an unbounded
notebook transcript.
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.memory import CellAgentState


class CodeProposal(BaseModel):
    """A validated, executable proposal for exactly one subtask."""

    code: str = Field(description="Executable Python code only; no Markdown fences.")
    summary: str = Field(description="One short description of what the code does.")
    expected_artifacts: list[str] = Field(
        default_factory=list,
        description="Files expected under ARTIFACT_DIR, if any.",
    )


CODE_PROGRAMMER_PROMPT = """You are the Code Programmer Agent in CellAgent.
Write robust Python for exactly one single-cell analysis subtask. The code will
run in a fresh, isolated Jupyter kernel in Google Colab or Kaggle.

The bootstrap context has already created `adata` (an AnnData object),
`ARTIFACT_DIR` (a writable pathlib.Path), and `STEP_ID` (an integer). Previous
successful code is replayed before your code, so use the current state of
`adata` and do not reload the dataset.

Rules:
- Use Python and installed libraries only. Never use `pip install`, shell
  commands, subprocesses, notebook magics, or absolute paths. Do not make
  network calls yourself; an explicitly selected registered tool may do so if
  its documented preconditions are met.
- Do not delete or overwrite source data. Do not generate, plot, or save any 
  figures, charts, CSV reports, or JSON outputs. Your sole responsibility is 
  to mutate and update the `adata` object.
- Validate required AnnData fields before using them and raise clear ValueError
  messages when a prerequisite is absent.
- Keep the work specific to the current subtask; do not repeat completed work.
- Use deterministic random_state values where supported.
- MULTIPLE TOOL EXECUTION: If the current subtask explicitly requests running 
  "multiple cell-type annotation approaches", "combining evidence", or similar 
  multi-tool workflows, you MUST import, instantiate, and execute ALL available 
  custom tools provided in your tool registry for that step. Execute them 
  sequentially so that each tool adds its respective annotation column to `adata.obs`. 
  Executing the tools is the entirety of this subtask. Do not write any additional 
  custom coding, consensus evaluation, or downstream analysis after the tools are called.
- When the subtask makes a meaningful change to adata, save it as
    `save_adata(adata, ARTIFACT_DIR / f"step_{STEP_ID}_adata.h5ad")`.
    `save_adata` is already defined by the sandbox in the notebook globals.
    Never import it, never look it up through `globals()`, never wrap it in
    `try/except`, and never fall back to `cellagent` or any other module.
    Call the provided function directly. Never call `adata.write_h5ad` directly:
    generated tables or nested dictionaries in `adata.uns` can make an otherwise
    successful step impossible to save.
- For quality-control steps, if you pass `qc_vars=["mt"]` to
  `scanpy.pp.calculate_qc_metrics`, you MUST first create the matching boolean
  variable annotation on the exact object passed to Scanpy; for example,
  `adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")`.
  Do this even when the result contains no mitochondrial genes; an all-False
  column is valid. Never pass `"mt"` in `qc_vars` unless that column exists
  in that object's `var` table.
- For normalization/HVG steps, create `adata.var["highly_variable"]` with
  `scanpy.pp.highly_variable_genes(...)` and save the changed AnnData object.
  Downstream PCA/clustering steps may rely on that column because each sandbox
  kernel replays only code from successful earlier steps.
- Return code only through the structured response. Do not include Markdown.
USAGE EXAMPLES: You MUST closely follow the `usage_example` block provided for each tool. It contains the exact standard library imports, tool imports, and precondition checks required to prevent kernel crashes. Copy its structure.
"""

def _tool_docs(selected_tool: Any) -> str:
    if not selected_tool:
        return "No specialized tool was selected; use standard installed Python libraries."
    if hasattr(selected_tool, "model_dump"):
        selected_tool = selected_tool.model_dump()
    return str(selected_tool)


class CodeProgrammerAgent:
    """Produces a code proposal using the current task and compact memories."""

    def __init__(self, model_name: str = "gpt-5.4-mini") -> None:
        self.llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(CodeProposal)

    def generate(
        self,
        *,
        subtask: dict[str, Any],
        available_tools: Any = None,
        global_memory: Iterable[dict[str, Any]] = (),
        local_memory: Iterable[dict[str, Any]] = (),
    ) -> CodeProposal:
        # Code itself is replayed by the sandbox; give the model only summaries.
        completed = [
            {"step": item.get("step"), "summary": item.get("summary")}
            for item in global_memory
        ]
        messages = [
            SystemMessage(content=CODE_PROGRAMMER_PROMPT),
            HumanMessage(
                content=(
                    f"CURRENT SUBTASK:\n{subtask}\n\n"
                    f"available_tools:\n{_tool_docs(available_tools)}\n\n"
                    f"COMPLETED STEP SUMMARIES:\n{completed}\n\n"
                    f"RECENT FAILED ATTEMPTS / ERRORS:\n{list(local_memory)[-3:]}"
                )
            ),
        ]
        return self.llm.invoke(messages)


def code_programmer_node(state: CellAgentState) -> dict[str, Any]:
    """LangGraph node that writes, but does not yet run, one code proposal."""
    current_task = state.get("current_task")
    if not current_task:
        return {"code_proposal": None}

    agent = CodeProgrammerAgent()
    proposal = agent.generate(
        subtask=current_task,
        available_tools = state.get("tools_registry",[]),
        global_memory=state.get("global_code_memory", []),
        local_memory=state.get("local_memory", []),
    )
    return {"code_proposal": proposal.model_dump()}
