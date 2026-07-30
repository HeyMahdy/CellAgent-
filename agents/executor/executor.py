import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# 1. Output Schema for Tool Selector
# ==========================================
class ToolSelectionOutput(BaseModel):
    selected_tools: List[str] = Field(
        description="List of exact tool names chosen from the registry. Empty list if no tools are needed."
    )

# ==========================================
# 2. Tool Selector Agent (A^t_LLM)
# ==========================================
class ToolSelectorAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        
        system_prompt = """You are the Tool Selector Agent for CellAgent.
Your task is to analyze the current subtask and select the appropriate bioinformatics tools from the registered tool library.

Available Tools:
{available_tools_metadata}

Rules:
1. Select ONLY the exact tool names from the provided list.
2. If the subtask is a general data manipulation step (e.g., standard scanpy filtering or basic pandas operations) that does not require a specialized registered tool, return an empty list.
3. Only select tools that directly address the subtask description.
"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Subtask to complete: {subtask_title}\nDescription: {subtask_description}")
        ])
        
        self.chain = self.prompt | self.llm.with_structured_output(ToolSelectionOutput)

    def select_tools(self, subtask: Dict[str, Any], available_tools: List[Dict[str, str]]) -> List[str]:
        """Evaluates the subtask and returns a list of selected tool names."""
        print(f"🔍 Tool Selector evaluating Subtask: {subtask['title']}...")
        
        # Format the available tools for the prompt
        formatted_tools = ""
        for tool in available_tools:
            formatted_tools += f"- {tool['name']}: {tool['description']}\n"
            
        if not formatted_tools:
            formatted_tools = "No specialized tools registered."

        response: ToolSelectionOutput = self.chain.invoke({
            "available_tools_metadata": formatted_tools,
            "subtask_title": subtask["title"],
            "subtask_description": subtask["description"]
        })
        
        print(f"  └── Selected Tools: {response.selected_tools}")
        return response.selected_tools


# ==========================================
# 3. Code Programmer Agent (A^c_LLM)
# ==========================================
class CodeProgrammerAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        
        self.system_template = """You are the Code Programmer Agent for CellAgent.
Your job is to write pure, executable Python (or R via Python subprocess) code to complete the given subtask.

--- Global Code Memory ($M$) ---
This is the successful code executed in previous steps. Assume variables created here (like `adata`) are available in your namespace. Do NOT re-execute these steps.
{global_context}

--- Selected Tools Documentation ---
{tool_docs}

Rules:
1. Output ONLY executable Python code.
2. Put the code inside a single ```python ... ``` markdown block.
3. Include brief inline comments explaining the logic.
4. If an exception occurred in a previous attempt, fix the exact error shown in the local memory.
5. If generating a plot, ensure you call `plt.show()` so the Sandbox can capture the image.
"""

    def generate_code(
        self, 
        subtask: Dict[str, Any], 
        tool_docs: str, 
        global_context: str, 
        local_messages: List[Dict[str, str]]
    ) -> str:
        """Generates code based on the subtask, tools, memory, and any previous error feedback."""
        print(f"💻 Code Programmer writing code for Subtask: {subtask['title']}...")
        
        # 1. Construct the System Message
        sys_msg_content = self.system_template.format(
            global_context=global_context,
            tool_docs=tool_docs
        )
        messages = [SystemMessage(content=sys_msg_content)]
        
        # 2. Append the initial task instruction
        task_msg = f"Write the code to complete this subtask:\nTitle: {subtask['title']}\nDescription: {subtask['description']}"
        messages.append(HumanMessage(content=task_msg))
        
        # 3. Append Local Memory (the trial-and-error exception loop history)
        for msg in local_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        # 4. Invoke LLM
        response = self.llm.invoke(messages)
        
        # 5. Extract just the Python code block
        return self._extract_code_block(response.content)
        
    def _extract_code_block(self, raw_text: str) -> str:
        """Regex helper to extract code from inside ```python ... ``` blocks."""
        match = re.search(r"```python\n(.*?)\n```", raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: if the LLM didn't use markdown fences, return the raw text
        return raw_text.strip()
