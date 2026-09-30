#!/usr/bin/env python3
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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Directory for user-generated files (PDFs, exports, etc.)
USER_DOWNLOADS_DIR: Path = Path("user_downloads")


# -----------------------------------------------------------------------------
# Tool Implementations
# -----------------------------------------------------------------------------


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


def save_summary_to_pdf(filename: str = "", content: str = "", **kwargs) -> str:
    """
    Saves text content to a PDF file on the user's disk.

    Use this tool when the user asks to:
    - "Save this as a PDF"
    - "Create a file with this content"
    - "Export this summary"
    - "Generate a document"

    Args:
        filename: The name for the PDF file (without .pdf extension).
        content: The text content to save in the PDF.

    Returns:
        A success message with the file path, or an error message.
    """
    # Handle dict args
    if isinstance(filename, dict):
        content = filename.get("content", "")
        filename = filename.get("filename", "document")
    
    try:
        if not filename or not str(filename).strip():
            filename = "document"
        
        filename = str(filename).strip()

        if not content or not str(content).strip():
            return "Error: Content cannot be empty."
        
        content = str(content)

        # Sanitize filename
        safe_filename = "".join(
            c for c in filename if c.isalnum() or c in (" ", "-", "_")
        ).strip()

        if not safe_filename:
            safe_filename = "document"

        if not safe_filename.lower().endswith(".pdf"):
            safe_filename += ".pdf"

        # Create downloads directory
        USER_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = USER_DOWNLOADS_DIR / safe_filename

        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 16)
        title = safe_filename.replace(".pdf", "").replace("_", " ").title()
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(10)

        # Content
        pdf.set_font("Arial", size=11)
        safe_content = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 8, safe_content)

        # Footer
        pdf.ln(10)
        pdf.set_font("Arial", "I", 9)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 10, f"Generated on: {timestamp}", ln=True, align="R")

        pdf.output(str(output_path))

        return (
            f"PDF saved successfully!\n"
            f"  Filename: {safe_filename}\n"
            f"  Location: {output_path.absolute()}\n"
            f"  Size: {os.path.getsize(output_path)} bytes"
        )

    except Exception as e:
        return f"Error: Failed to save PDF - {str(e)}"


def fetch_coding_guidelines(language: str = "", **kwargs) -> str:
    """
    Retrieves style guides and best practices for a programming language.

    Use this tool when:
    - The user asks for code in a specific language
    - You need to write production-quality code
    - The user mentions coding standards or style guides

    Supported languages: Python, C, C++, JavaScript, TypeScript, Rust, Go, Java

    Args:
        language: The programming language name.

    Returns:
        A string containing the style guide for the language.
    """
    # Handle dict args
    if isinstance(language, dict):
        language = language.get("language", "") or language.get("input", "")
    
    try:
        if not language or not str(language).strip():
            return "Error: Language parameter cannot be empty."

        lang_lower = str(language).strip().lower()

        guidelines = {
            "python": (
                "PYTHON STYLE GUIDE (PEP 8 Standard):\n"
                "• Naming: Use snake_case for functions/variables, PascalCase for classes\n"
                "• Type Hints: Always include type annotations for function parameters and returns\n"
                "• Docstrings: Use triple-quoted strings for all public functions and classes\n"
                "• Line Length: Maximum 88 characters (Black formatter standard)\n"
                "• Imports: Group in order - stdlib, third-party, local; use absolute imports\n"
                "• Error Handling: Use specific exception types, not bare 'except:'\n"
                "• f-strings: Prefer f-strings over .format() or % formatting"
            ),
            "c": (
                "EMBEDDED C STYLE GUIDE:\n"
                "• Memory: Manage memory manually; always free() allocated memory\n"
                "• Null Checks: Validate all pointers before dereferencing\n"
                "• Header Guards: Use #ifndef/#define/#endif in all header files\n"
                "• Naming: Use snake_case, prefix functions with module name\n"
                "• Constants: Use #define or enum for constants, avoid magic numbers\n"
                "• Comments: Document complex algorithms and non-obvious logic\n"
                "• Error Codes: Return error codes from functions; check all return values"
            ),
            "cpp": (
                "C++ STYLE GUIDE (Modern C++17/20):\n"
                "• Smart Pointers: Use unique_ptr/shared_ptr instead of raw pointers\n"
                "• RAII: Use Resource Acquisition Is Initialization pattern\n"
                "• Naming: PascalCase for classes, camelCase for methods, snake_case for variables\n"
                "• Const Correctness: Mark functions and parameters const when appropriate\n"
                "• STL: Prefer STL containers and algorithms over raw arrays\n"
                "• Move Semantics: Use std::move for transferring ownership\n"
                "• Avoid: Macros (prefer constexpr), raw new/delete, C-style casts"
            ),
            "javascript": (
                "JAVASCRIPT STYLE GUIDE (ESLint/Airbnb Standard):\n"
                "• Variables: Use const by default, let when reassignment needed, never var\n"
                "• Functions: Prefer arrow functions for callbacks, named for top-level\n"
                "• Naming: camelCase for variables/functions, PascalCase for classes/components\n"
                "• Async: Use async/await over raw Promises or callbacks\n"
                "• Destructuring: Use for object/array access when appropriate\n"
                "• Modules: Use ES6 import/export syntax\n"
                "• Equality: Always use === and !== (strict equality)"
            ),
            "typescript": (
                "TYPESCRIPT STYLE GUIDE:\n"
                "• Types: Define explicit types; avoid 'any' except when truly necessary\n"
                "• Interfaces: Prefer interfaces over type aliases for object shapes\n"
                "• Enums: Use string enums for better debugging; consider const enums\n"
                "• Generics: Use meaningful names (T for type, K for key, V for value)\n"
                "• Null Safety: Enable strict null checks; use optional chaining (?.)\n"
                "• Return Types: Always specify function return types explicitly\n"
                "• Readonly: Use readonly for immutable properties"
            ),
            "rust": (
                "RUST STYLE GUIDE (rustfmt Standard):\n"
                "• Ownership: Understand borrowing rules; prefer references over cloning\n"
                "• Naming: snake_case for functions/variables, PascalCase for types\n"
                "• Error Handling: Use Result<T, E> and Option<T>; avoid unwrap() in prod\n"
                "• Pattern Matching: Use match expressions for exhaustive handling\n"
                "• Clippy: Follow all clippy lints for idiomatic code\n"
                "• Lifetimes: Minimize explicit lifetime annotations; let compiler infer\n"
                "• Traits: Implement standard traits (Debug, Clone, etc.) where appropriate"
            ),
            "go": (
                "GO STYLE GUIDE (gofmt/Effective Go):\n"
                "• Formatting: Use gofmt; it's non-negotiable\n"
                "• Naming: MixedCase (exported), mixedCase (unexported); short names in scope\n"
                "• Errors: Return errors as last value; check errors explicitly\n"
                "• Interfaces: Keep small; define at point of use\n"
                "• Goroutines: Always handle goroutine cleanup; use context for cancellation\n"
                "• Comments: Document all exported items; start with the name\n"
                "• Packages: One package per directory; meaningful, short names"
            ),
            "java": (
                "JAVA STYLE GUIDE (Google Java Style):\n"
                "• Naming: PascalCase for classes, camelCase for methods/variables\n"
                "• Braces: K&R style (opening brace on same line)\n"
                "• Immutability: Prefer final for variables; use immutable collections\n"
                "• Streams: Use Java 8+ streams for collection operations\n"
                "• Optional: Use Optional<T> instead of null for optional returns\n"
                "• Exceptions: Use checked exceptions sparingly; prefer unchecked\n"
                "• Javadoc: Document all public APIs with @param, @return, @throws"
            ),
        }

        # Handle c++ alias
        if lang_lower == "c++":
            lang_lower = "cpp"

        if lang_lower in guidelines:
            return guidelines[lang_lower]
        else:
            supported = ", ".join(sorted(set(k for k in guidelines.keys())))
            return (
                f"No specific guidelines found for '{language}'.\n"
                f"Supported languages: {supported}\n\n"
                "General best practices:\n"
                "• Use meaningful, descriptive names\n"
                "• Keep functions small and focused\n"
                "• Write clear comments for complex logic\n"
                "• Handle errors gracefully\n"
                "• Follow DRY (Don't Repeat Yourself) principle"
            )

    except Exception as e:
        return f"Error: Could not fetch coding guidelines - {str(e)}"


def activate_teacher_mode(topic: str = "", **kwargs) -> str:
    """
    Activates 'Teacher Mode' for guided, Socratic-style learning.

    Use this tool when the user asks to:
    - "Teach me about X"
    - "Help me understand X"
    - "Explain X like I'm a beginner"
    - "I want to learn about X"

    Args:
        topic: The subject or concept the user wants to learn about.

    Returns:
        An instruction string that guides Socratic tutoring behavior.
    """
    # Handle dict args
    if isinstance(topic, dict):
        topic = topic.get("topic", "") or topic.get("input", "")
    
    try:
        if not topic or not str(topic).strip():
            return "Error: Topic cannot be empty. Please specify what you want to learn."

        clean_topic = str(topic).strip()

        return (
            f"INSTRUCTION: You are now a Socratic Tutor for '{clean_topic}'.\n\n"
            f"BEHAVIOR GUIDELINES:\n"
            f"• Do NOT give answers directly\n"
            f"• Ask guiding questions to help the user discover insights\n"
            f"• Break complex concepts into smaller, digestible pieces\n"
            f"• Validate the user's understanding before moving forward\n"
            f"• Use analogies and real-world examples\n"
            f"• Encourage the user when they make progress\n"
            f"• If the user is stuck, provide gentle hints, not solutions\n\n"
            f"START by asking the user what they already know about '{clean_topic}' "
            f"to gauge their current understanding level."
        )

    except Exception as e:
        return f"Error: Could not activate teacher mode - {str(e)}"


def evaluate_math_expression(expression: str = "", **kwargs) -> str:
    """
    Safely evaluates a mathematical expression and returns the result.

    Use this tool when the user needs:
    - Basic arithmetic calculations
    - Mathematical computations
    - Number conversions or calculations

    Supports: +, -, *, /, **, %, parentheses.

    Args:
        expression: A mathematical expression string (e.g., "2 + 2 * 3").

    Returns:
        The computed result as a string, or an error message.
    """
    # Handle dict args
    if isinstance(expression, dict):
        expression = expression.get("expression", "") or expression.get("input", "")
    
    try:
        if not expression or not str(expression).strip():
            return "Error: Expression cannot be empty."

        expression = str(expression).strip()
        result = simple_eval(expression)

        return f"Result: {expression} = {result}"

    except Exception as e:
        return f"Error: Could not evaluate expression - {str(e)}"


def update_core_memory(section: str = "", field: str = "", value: str = "", operation: str = "replace", **kwargs) -> str:
    """
    Updates the core memory (User, Persona, or Context).

    Use this tool when you need to:
    - Update user details (name, preferences)
    - Modify persona traits
    - Update conversation context

    Args:
        section: The memory section ('user', 'persona', 'conversation_context').
        field: The specific field to update.
        value: The value to set.
        operation: 'replace' (default) or 'append' (for lists).

    Returns:
        Confirmation string for the memory engine.
    """
    # Handle dict args (in case agent passes a single dictionary object)
    if isinstance(section, dict):
        kwargs = section
        section = kwargs.get("section", "")
        field = kwargs.get("field", "")
        value = kwargs.get("value", "")
        operation = kwargs.get("operation", "replace")
    
    try:
        # Validate inputs
        if not section or not str(section).strip():
            return "Error: Section cannot be empty. Use 'user', 'persona', or 'conversation_context'."
        
        if not field or not str(field).strip():
            return "Error: Field cannot be empty."
        
        if not value or not str(value).strip():
            return "Error: Value cannot be empty."
        
        section = str(section).strip().lower()
        field = str(field).strip().lower()
        value = str(value).strip()
        operation = str(operation).strip().lower()

        valid_sections = ["user", "persona", "conversation_context"]
        if section not in valid_sections:
            return f"Error: Invalid section '{section}'. Must be one of {valid_sections}."
        
        # Return instructions for the Engine
        return (
            f"MEMORY_UPDATE_REQUEST:\n"
            f"  Section: {section}\n"
            f"  Field: {field}\n"
            f"  Value: {value}\n"
            f"  Operation: {operation}\n\n"
            f"System Notification: Requesting update to {section}.{field}..."
        )
        
    except Exception as e:
        return f"Error: Could not update memory - {str(e)}"


def search_knowledge_base(query: str = "", agent: Optional[Any] = None, **kwargs) -> str:
    """
    Searches the archival memory / knowledge base for relevant information.

    Use this tool when:
    - The user asks about something that might be in their documents
    - You need to recall previously stored information
    - Looking up facts from uploaded files

    Args:
        query: The search query.
        agent: Optional MemoryAgent instance to use for the search.
               If not provided, creates a new one (expensive, avoid if possible).

    Returns:
        Search results from the knowledge base.
    """
    # Handle dict args
    if isinstance(query, dict):
        query = query.get("query", "") or query.get("input", "")
    
    try:
        if not query or not str(query).strip():
            return "Error: Search query cannot be empty."
        
        query = str(query).strip()
        
        # Use provided agent or create new one
        if agent is None:
            # Import here to avoid circular dependency at module level
            try:
                from engine import MemoryAgent
            except ImportError:
                try:
                    from src.engine import MemoryAgent
                except ImportError:
                    return "Error: Could not import MemoryAgent. Check your installation."
            
            agent = MemoryAgent()
        
        results = agent.query_archival_memory(query, top_k=5)
        
        if not results:
            return f"No relevant documents found for: '{query}'"
        
        formatted = [f"Knowledge Base Search Results for '{query}':\n"]
        for i, doc in enumerate(results, 1):
            formatted.append(f"\n--- Result {i} ---\n{doc}")
        
        return "\n".join(formatted)
        
    except Exception as e:
        return f"Error: Knowledge base search failed - {str(e)}"


# -----------------------------------------------------------------------------
# Tool Registry and Dispatcher
# -----------------------------------------------------------------------------


# Registry of all available tools with metadata
AVAILABLE_TOOLS: dict[str, dict[str, Any]] = {
    "perform_web_search": {
        "function": perform_web_search,
        "description": "Search the internet for real-time information, news, or facts",
        "parameters": ["query"],
    },
    "get_current_datetime": {
        "function": get_current_datetime,
        "description": "Get the current date and time",
        "parameters": [],
    },
    "save_summary_to_pdf": {
        "function": save_summary_to_pdf,
        "description": "Save text content to a PDF file",
        "parameters": ["filename", "content"],
    },
    "fetch_coding_guidelines": {
        "function": fetch_coding_guidelines,
        "description": "Get coding style guidelines for a programming language",
        "parameters": ["language"],
    },
    "activate_teacher_mode": {
        "function": activate_teacher_mode,
        "description": "Activate Socratic teaching mode for guided learning",
        "parameters": ["topic"],
    },
    "evaluate_math_expression": {
        "function": evaluate_math_expression,
        "description": "Safely evaluate a mathematical expression",
        "parameters": ["expression"],
    },
    "update_core_memory": {
        "function": update_core_memory,
        "description": "Update core memory sections (User, Persona, Context)",
        "parameters": ["section", "field", "value", "operation"],
    },
    "search_knowledge_base": {
        "function": search_knowledge_base,
        "description": "Search the knowledge base for relevant documents",
        "parameters": ["query"],
    },
}


def run_tool(tool_name: str, args: dict[str, Any]) -> str:
    """
    Execute a tool by name with the given arguments.

    Args:
        tool_name: Name of the tool to execute.
        args: Dictionary of arguments to pass to the tool.

    Returns:
        String result from the tool execution.
    """
    if tool_name not in AVAILABLE_TOOLS:
        available = ", ".join(AVAILABLE_TOOLS.keys())
        return f"Error: Unknown tool '{tool_name}'. Available tools: {available}"

    tool_info = AVAILABLE_TOOLS[tool_name]
    tool_func = tool_info["function"]

    try:
        # Call the tool with the provided arguments
        result = tool_func(**args)
        return str(result)
    except TypeError as e:
        # Handle missing required arguments
        return f"Error: Invalid arguments for '{tool_name}' - {str(e)}"
    except Exception as e:
        return f"Error: Tool '{tool_name}' failed - {str(e)}"


def list_tools() -> str:
    """List all available tools with their descriptions."""
    lines = ["Available Tools:"]
    for name, info in AVAILABLE_TOOLS.items():
        params = ", ".join(info["parameters"]) if info["parameters"] else "none"
        lines.append(f"\n• {name}({params})")
        lines.append(f"  {info['description']}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI Testing
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("TOOL TESTING")
    print("=" * 60)
    print(list_tools())
    
    print("\n" + "-" * 60)
    print("Testing tools...")
    print("-" * 60)
    
    # Test get_current_datetime
    print("\n1. get_current_datetime:")
    print(run_tool("get_current_datetime", {}))
    
    # Test evaluate_math_expression
    print("\n2. evaluate_math_expression:")
    print(run_tool("evaluate_math_expression", {"expression": "2 + 2 * 3"}))
    
    # Test fetch_coding_guidelines
    print("\n3. fetch_coding_guidelines:")
    print(run_tool("fetch_coding_guidelines", {"language": "python"}))
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

    # Test update_core_memory (New Tool)
    print("\n4. update_core_memory:")
    print(run_tool("update_core_memory", {
        "section": "persona",
        "field": "traits",
        "value": "patient",
        "operation": "append"
    }))
