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
