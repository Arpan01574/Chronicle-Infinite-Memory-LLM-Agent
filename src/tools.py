"""
Custom Tools for Infinite Memory LLM Agent.

This module defines tools that can be invoked by the Memory Agent.
Each tool is a simple function that takes arguments and returns a string result.

Tools:
    1. perform_web_search - Real-time internet search via DuckDuckGo
    2. get_current_datetime - Current date/time retrieval
    3. save_summary_to_pdf - Export text content to PDF files
    4. fetch_coding_guidelines - Language-specific coding standards
    5. activate_teacher_mode - Socratic tutoring activation
    6. evaluate_math_expression - Safe math evaluation
    7. update_core_memory - Update core memory about the user
    8. search_knowledge_base - Search the archival memory

Usage:
    from src.tools import run_tool, AVAILABLE_TOOLS
    
    result = run_tool("perform_web_search", {"query": "latest AI news"})
    print(result)
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from duckduckgo_search import DDGS
from fpdf import FPDF
from simpleeval import simple_eval
USER_DOWNLOADS_DIR: Path = Path("user_downloads")
def perform_web_search(query: str = "", **kwargs) -> str:
    """
    Searches the internet for real-time information, news, or facts.

    Use this tool when the user asks about:
    - Current events, news, or recent developments
    - Real-time data (weather, stock prices, sports scores)
    - Facts you're uncertain about or that may have changed
    - Topics that require up-to-date information

    Args:
        query: The search query string to look up on the internet.

    Returns:
        A summarized string containing the top 3 search results.
    """
    # Handle both direct string and dict args
    if isinstance(query, dict):
        query = query.get("query", "") or query.get("input", "")
    
    try:
        if not query or not str(query).strip():
            return "Error: Search query cannot be empty."

        query = str(query).strip()

        # Initialize DuckDuckGo Search client
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return f"No search results found for: '{query}'"

        # Format results into a readable summary
        formatted_results = []
        for i, result in enumerate(results, start=1):
            title = result.get("title", "No Title")
            body = result.get("body", "No description available.")
            href = result.get("href", "")

            formatted_results.append(
                f"**Result {i}:** {title}\n"
                f"  Summary: {body}\n"
                f"  Source: {href}"
            )

        return (
            f"Web Search Results for '{query}':\n\n"
            + "\n\n".join(formatted_results)
        )

    except Exception as e:
        return f"Error: Web search failed - {str(e)}"
def get_current_datetime(**kwargs) -> str:
    """
    Returns the current date and time.

    Use this tool when the user asks about:
    - "What time is it?"
    - "What's today's date?"
    - Any query requiring knowledge of the current moment
    - Scheduling or time-based calculations

    Returns:
        A formatted string showing the current date, time, and day of week.
    """
    try:
        now = datetime.now()
        formatted = now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
        iso_format = now.isoformat()

        return (
            f"Current Date and Time:\n"
            f"  Formatted: {formatted}\n"
            f"  ISO 8601:  {iso_format}\n"
            f"  Timezone:  Local System Time"
        )

    except Exception as e:
        return f"Error: Could not retrieve current time - {str(e)}"
