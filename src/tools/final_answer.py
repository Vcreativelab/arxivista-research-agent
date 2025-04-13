# It Combines results from ArXiv, RAG, and web search into a final response.

from typing import Union, List

def final_answer(
        introduction: str,
        research_steps: Union[str, List[str]],
        main_body: str,
        conclusion: str,
        sources: Union[str, List[str]]
) -> str:
    """
    Generates a research report as a plain text string.
    This function is used as a tool in the agent pipeline.

    Args:
        introduction (str): Introduction text.
        research_steps (Union[str, List[str]]): Research steps; if a list, each step is prefixed with '- '.
        main_body (str): The main content of the report.
        conclusion (str): The conclusion text.
        sources (Union[str, List[str]]): Sources; if a list, each source is prefixed with '- '.

    Returns:
        str: The combined research report.
    """
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])
    return (
        f"{introduction}\n\n"
        f"Research Steps:\n{research_steps}\n\n"
        f"Main Body:\n{main_body}\n\n"
        f"Conclusion:\n{conclusion}\n\n"
        f"Sources:\n{sources}"
    )


def format_final_answer(output: dict) -> str:
    """
    Formats the final answer output into a nicely designed report for display.
    Expects a dictionary with keys: introduction, research_steps, main_body, conclusion, sources.

    Returns:
        str: A formatted multi-section report.
    """
    research_steps = output.get("research_steps", [])
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])
    sources = output.get("sources", [])
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])
    return f"""
INTRODUCTION
------------
{output.get("introduction", "N/A")}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output.get("main_body", "N/A")}

CONCLUSION
----------
{output.get("conclusion", "N/A")}

SOURCES
-------
{sources}
"""
