# src/tools/final_answer.py

from typing import Union, List, Dict


def final_answer(
    introduction: str,
    research_steps: Union[str, List[str]],
    main_body: str,
    conclusion: str,
    sources: Union[str, List[str]]
) -> Dict:
    """
    Tool used by the agent to produce a structured final answer.
    Returns a dictionary matching the unified tool schema.
    """

    if isinstance(research_steps, list):
        research_steps = [str(r) for r in research_steps]

    if isinstance(sources, list):
        sources = [str(s) for s in sources]

    return {
        "success": True,
        "introduction": introduction,
        "research_steps": research_steps,
        "main_body": main_body,
        "conclusion": conclusion,
        "sources": sources,
    }


def format_final_answer(output: Dict) -> str:
    """
    Convert the oracle final_answer tool output into a formatted markdown report.
    """

    intro = output.get("introduction", "N/A")

    steps = output.get("research_steps", [])
    if isinstance(steps, list):
        steps = "\n".join([f"- {s}" for s in steps])

    main_body = output.get("main_body", "N/A")
    conclusion = output.get("conclusion", "N/A")

    sources = output.get("sources", [])
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])

    return f"""
INTRODUCTION
------------
{intro}

RESEARCH STEPS
--------------
{steps}

REPORT
------
{main_body}

CONCLUSION
----------
{conclusion}

SOURCES
-------
{sources}
"""
