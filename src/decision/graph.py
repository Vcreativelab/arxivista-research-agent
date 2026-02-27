# src/decision/graph.py
# Decision graph for routing oracle → tools → oracle → final_answer.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from typing import List, TypedDict, Annotated, Dict
import operator
import json


# ---------------- Agent State ----------------
class AgentState(TypedDict):
    input: str
    messages: List[BaseMessage]

    # executed tools ONLY
    intermediate_steps: Annotated[List[AgentAction], operator.add]

    # execution guards
    tool_usage: Dict[str, int]

    # oracle planning output
    next_tool: str
    next_tool_args: Dict


# ---------------- Import oracle and tools ----------------
from src.decision.oracle import oracle
from src.tools.rag_search_filter import rag_search_filter
from src.tools.rag_search import rag_search
from src.tools.fetch_arxiv import fetch_arxiv
from src.tools.web_search import web_search
from src.tools.final_answer import final_answer


# ---------------- Execution Guards ----------------
MAX_STEPS = 6
MAX_TOOL_USAGE = 1


# ---------------- Run Oracle ----------------
def run_oracle(state: dict) -> dict:
    """
    Executes the LLM oracle responsible for selecting
    the next tool to execute.

    This step performs PLANNING only.
    No intermediate_steps are created here.
    """

    out = oracle.invoke(state)

    tool_call = out.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"\n🧭 ORACLE → tool: {tool_name}")

    usage = state.get("tool_usage", {})
    usage[tool_name] = usage.get(tool_name, 0) + 1

    return {
        "next_tool": tool_name,
        "next_tool_args": tool_args,
        "tool_usage": usage
    }


# ---------------- Router Logic ----------------
def router(state: dict) -> str:
    """
    Determines which node executes next.

    Enforces:
    - global recursion guard
    - safe oracle routing
    """

    steps = state.get("intermediate_steps", [])

    if len(steps) >= MAX_STEPS:
        print("⚠️ Max execution steps reached")
        return "final_answer"

    next_tool = state.get("next_tool")

    if not next_tool:
        print("⚠️ Missing oracle decision")
        return "final_answer"

    return next_tool


# ---------------- Tool Execution ----------------
tool_str_to_func = {
    "rag_search_filter": rag_search_filter,
    "rag_search": rag_search,
    "fetch_arxiv": fetch_arxiv,
    "web_search": web_search,
    "final_answer": final_answer
}


def run_tool(state: dict) -> dict:
    """
    Executes the oracle-selected tool and records
    the completed execution into intermediate_steps.
    """

    tool_name = state["next_tool"]
    tool_args = state["next_tool_args"]

    usage = state.get("tool_usage", {})

    if usage.get(tool_name, 0) > MAX_TOOL_USAGE:
        print(f"⚠️ Tool {tool_name} exceeded usage")
        tool_name = "final_answer"
        tool_args = {"error": "Tool usage exceeded"}

    tool_func = tool_str_to_func[tool_name]

    print(f"🔧 TOOL EXECUTION → {tool_name}")

    result = tool_func(**tool_args)

    return {
        "intermediate_steps": [
            AgentAction(
                tool=tool_name,
                tool_input=tool_args,
                log=json.dumps(result, default=str)
            )
        ]
    }


# ---------------- Build Graph ----------------
graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)

for tool in tool_str_to_func:
    graph.add_node(tool, run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges("oracle", router)

for tool in ["rag_search_filter", "rag_search", "fetch_arxiv", "web_search"]:
    graph.add_edge(tool, "oracle")

graph.add_edge("final_answer", END)

runnable = graph.compile()
