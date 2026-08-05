"""Notebook-isolated code execution for Colab, Kaggle, and local Jupyter.

This is an execution isolation boundary, not a security boundary. Only run
generated code for trusted users/datasets in a runtime you control.
"""

from __future__ import annotations
import scanpy as sc
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from pathlib import Path
from dataclasses import asdict
import scanpy as sc
import os
import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


@dataclass(slots=True)
class SandboxResult:
    success: bool
    stdout: str
    error: str | None
    traceback: str | None
    notebook_path: str
    artifacts: list[str]


def default_artifact_dir() -> Path:
    """Choose a writable, persistent location for hosted notebook platforms."""
    if Path("/content").is_dir():
        return Path("/content/cellagent_artifacts")
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/cellagent_artifacts")
    return Path.cwd() / "cellagent_artifacts"


class NotebookSandbox:
    """Run a candidate cell in a new kernel while replaying successful steps."""

    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        timeout_seconds: int = 900,
        kernel_name: str = "python3",
        project_root: str | Path | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir else default_artifact_dir()
        self.timeout_seconds = timeout_seconds
        self.kernel_name = kernel_name
        self.project_root = Path(project_root) if project_root else Path.cwd()

    def execute(
        self,
        code: str,
        *,
        dataset_path: str,
        step_id: int = 0,
    ) -> SandboxResult:
        """Execute candidate code and return serializable output/error details.

        ``bootstrap_code`` must establish `adata`; e.g. it can read a source
        .h5ad file in the Colab/Kaggle runtime. Successful code is replayed in
        order because a fresh notebook kernel is used for every attempt.
        """
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        setup = (
          "import sys\n"
            "import scanpy as sc\n"
            f"PROJECT_ROOT = {str(self.project_root)!r}\n"
            "if PROJECT_ROOT not in sys.path:\n"
            "    sys.path.insert(0, PROJECT_ROOT)\n"
            "from pathlib import Path\n"
            f"ARTIFACT_DIR = Path({str(self.artifact_dir)!r})\n"
            "ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)\n"
            f"STEP_ID = {step_id}\n"
            "if STEP_ID > 1:\n"
            "    # Load previous step's artifact\n"
            "    adata = sc.read_h5ad(ARTIFACT_DIR / f\"step_{STEP_ID - 1}_adata.h5ad\")\n"
            "else:\n"
            "    # Initial step: load raw dataset\n"
            "    from data.ingest_dataset_to_anndata import ingest_dataset_to_anndata\n"
            f"    adata = ingest_dataset_to_anndata({dataset_path!r})\n"
            "def save_adata(adata, path):\n"
            "    \"\"\"Save a portable analysis artifact without unsafe LLM metadata.\"\"\"\n"
            "    artifact = adata.copy()\n"
            "    artifact.write_h5ad(path)\n"
        )
        notebook = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell(setup)]
            + [nbformat.v4.new_code_cell(code)],
            metadata={"kernelspec": {"name": self.kernel_name, "display_name": "Python 3"}},
        )

        with tempfile.TemporaryDirectory(prefix="cellagent-sandbox-") as temp_dir:
            notebook_path = Path(temp_dir) / f"step_{step_id}.ipynb"
            nbformat.write(notebook, notebook_path)
            executor = ExecutePreprocessor(timeout=self.timeout_seconds, kernel_name=self.kernel_name)
            error: str | None = None
            trace: str | None = None
            try:
                executed, _ = executor.preprocess(notebook, {"metadata": {"path": temp_dir}})
                success = True
            except CellExecutionError as exc:
                executed = notebook
                success = False
                error = str(exc)
                trace = traceback.format_exc()
            except Exception as exc:  # Includes missing Jupyter kernels.
                executed = notebook
                success = False
                error = f"{type(exc).__name__}: {exc}"
                trace = traceback.format_exc()

            # Persist the notebook outside the temporary execution directory so
            # users can inspect the exact code and captured output in Colab/Kaggle.
            saved_notebook = self.artifact_dir / f"step_{step_id}_execution.ipynb"
            nbformat.write(executed, saved_notebook)
            stdout = _collect_text_output(executed)

        artifacts = sorted(
            str(path) for path in self.artifact_dir.iterdir() if path.name != saved_notebook.name
        )
        return SandboxResult(success, stdout, error, trace, str(saved_notebook), artifacts)


def _collect_text_output(notebook: nbformat.NotebookNode) -> str:
    parts: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                parts.append(output.get("text", ""))
            elif output.output_type == "error":
                parts.append("\n".join(output.get("traceback", [])))
            elif output.output_type in {"execute_result", "display_data"}:
                text = output.get("data", {}).get("text/plain")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def sandbox_node(state: dict) -> dict:
    """Optional LangGraph node to execute the pending `code_proposal`."""
    proposal = state.get("code_proposal")
    if not proposal:
        return {"sandbox_result": None}
        
    current_step = state.get("current_task_index", 0)
    
    # --- SAFE ARTIFACT DIR FALLBACK ---
    artifact_dir = state.get("artifact_dir")
    if not artifact_dir:
        artifact_dir = Path("cellagent_artifacts")
    else:
        artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # ----------------------------------
    
    sandbox = NotebookSandbox(
        artifact_dir=str(artifact_dir),
        timeout_seconds=state.get("sandbox_timeout_seconds", 900),
        project_root=state.get("project_root"),
    )
    
    result = sandbox.execute(
        proposal["code"],
        dataset_path=state["dataset_path"],
        step_id=current_step,
    )
    
    payload = asdict(result)
    
    if result.success:
        expected_artifact_path = artifact_dir / f"step_{current_step}_adata.h5ad"
        
        updated_adata = sc.read_h5ad(expected_artifact_path)
        # --- END SYNC LOGIC ---

        memory_item = {
            "step": current_step,
            "summary": proposal["summary"],
            "code": proposal["code"],
            "artifacts": result.artifacts,
        }
        
        return {
            "sandbox_result": payload, 
            "global_code_memory": [*state.get("global_code_memory", []), memory_item],
            "adata": updated_adata  # Overwrites the old adata in LangGraph!
        }
        
    return {
        "sandbox_result": payload,
        "local_memory": [
            *state.get("local_memory", []),
            {"step": current_step, "error": result.error, "stdout": result.stdout[-4000:]},
        ],
    }


def route_after_sandbox(state: dict) -> str:
    """Retry a failed code proposal twice, otherwise advance to the next task."""
    result = state.get("sandbox_result") or {}
    if result.get("success"):
        return "loop_node"

    step = state.get("current_task_index", 0)
    failures_for_step = sum(
        1 for item in state.get("local_memory", []) if item.get("step") == step
    )
    return "code_programmer_node" if failures_for_step < 2 else "loop_node"
