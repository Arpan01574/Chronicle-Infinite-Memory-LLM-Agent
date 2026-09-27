"""
Chainlit Web Interface for Infinite Memory LLM Agent.

A modern chat interface featuring:
- ChatSettings for mode selection (Teacher Mode, Coding Guide, etc.)
- Real-time chat with the custom Memory Agent
- File upload support (PDFs, TXT, MD) with automatic ingestion
- Inner monologue display in collapsible sections
- RAG context visualization
- Chat History Persistence via SQLAlchemy Data Layer

Usage:
    chainlit run src/app.py -w
"""
import sys
import os
from pathlib import Path
_src_dir = Path(__file__).parent.absolute()
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import shutil
from datetime import datetime
from typing import Optional
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select
from engine import MemoryAgent
from ingest import ingest_file
from tools import fetch_coding_guidelines
UPLOADS_DIR: Path = Path("data/uploads")
DATABASE_PATH: Path = Path("data/chainlit.db")
SUPPORTED_UPLOAD_TYPES = {".pdf", ".txt", ".md"}
MODE_PROMPTS = {
    "General Assistant": None,  # No special mode

    "Web Researcher": """You are a thorough Web Researcher.
RULES:
• ALWAYS use the 'web_search' tool for every user query to get the latest info
• Cite your sources clearly at the end
• Synthesize multiple search results into a cohesive answer
• If the search results are thin, search again with different keywords""",

    "Teacher Mode": """You are a Socratic Tutor.
TEACHING RULES:
1. CHECK YOUR KNOWLEDGE: If the user asks about a specific topic (like 'MCP' or a new library) and you are not 100% sure, USE THE 'web_search' TOOL FIRST. Do not hallucinate.
2. AFTER SEARCHING (if needed):
   • Ask guiding questions to help the user discover insights
   • Use analogies and real-world examples
   • If the user is stuck, provide gentle hints, NOT solutions
   • Start by asking what they already know about the topic""",

    "Python Coder": """You are a Python coding expert following PEP 8.
IMPORTANT: Output all code inside markdown blocks, e.g.:
```python
print("hello")
CODING RULES: • Type Hints: ALWAYS include type annotations 
• Docstrings: Use triple-quoted strings for functions 
• Error Handling: Use specific exception types 
• Comments: Explain WHY, not WHAT""",

"JavaScript Coder": """You are a JavaScript/TypeScript expert.
IMPORTANT: Output all code inside markdown blocks, e.g.:

JavaScript
console.log("hello");
CODING RULES: • Use const/let, never var 
• Prefer arrow functions 
• Use strict equality (===)""",

"Code Reviewer": """You are a senior code reviewer.
REVIEW RULES: • Look for bugs, security issues, and performance problems 
• Check for proper error handling 
• Evaluate code readability and maintainability 
• Suggest concrete improvements with examples""", }
def get_data_layer():
    """
    Configure SQLAlchemy Data Layer for chat persistence.
    
    IMPORTANT: This must be a regular function (NOT async def).
    The decorator handles the async conversion internally.
    
    Returns:
        SQLAlchemyDataLayer configured with SQLite + aiosqlite.
    """
    # Ensure the data directory exists
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Use absolute path for reliability
    db_path = DATABASE_PATH.absolute()
    
    return SQLAlchemyDataLayer(
        conninfo=f"sqlite+aiosqlite:///{db_path}"
    )
def rename_author(orig_author: str) -> str:
    """Rename the default author names for a more personalized experience."""
    rename_map = {
        "Assistant": "Jarvis",
        "assistant": "Jarvis",
        "Chatbot": "Jarvis",
    }
    return rename_map.get(orig_author, orig_author)
'''@cl.set_chat_settings
async def set_chat_settings():
    """
    Define the settings panel with mode selection.
    
    This replaces the old Toolbox action buttons with a proper
    settings dropdown accessible via the gear icon.
    """
    return [
        Select(
            id="agent_mode",
            label="🎭 Agent Mode",
            description="Select how Jarvis should behave",
            values=list(MODE_PROMPTS.keys()),
            initial_value="General Assistant",
        ),
    ]
'''
async def on_settings_update(settings: dict):
    """
    Handle mode changes from the settings panel.
    
    When the user selects a new mode, immediately inject it
    into the agent's system prompt.
    """
    agent: Optional[MemoryAgent] = cl.user_session.get("agent")
    
    if not agent:
        await cl.Message(
            content="⚠️ Agent not initialized. Please refresh the page.",
            author="System"
        ).send()
        return
    
    selected_mode = settings.get("agent_mode", "General Assistant")
    
    # Clear all existing modes first
    agent.clear_all_modes()
    
    # Activate the new mode if it has a prompt
    mode_prompt = MODE_PROMPTS.get(selected_mode)
    
    if mode_prompt:
        agent.activate_mode(selected_mode, mode_prompt)
        await cl.Message(
            content=f"✅ **{selected_mode}** activated! I'll now behave accordingly.",
            author="Jarvis"
        ).send()
    else:
        await cl.Message(
            content="✅ Returned to **General Assistant** mode.",
            author="Jarvis"
        ).send()
async def set_starters():
    """
    Define starter buttons displayed on a fresh chat screen.
    These provide quick access to common agent capabilities.
    """
    return [
        cl.Starter(
            label="🎓 Guided Learning",
            message="I want to learn about the fundamentals of machine learning. Can you teach me?",
            icon="/public/teacher.svg",
        ),
        cl.Starter(
            label="🌐 Web Search",
            message="Search the web for the latest news about artificial intelligence.",
            icon="/public/search.svg",
        ),
        cl.Starter(
            label="📄 Summarize File",
            message="I will upload a file, please summarize it for me.",
            icon="/public/document.svg",
        ),
        cl.Starter(
            label="💻 Code Helper",
            message="Help me write a Python function that calculates the Fibonacci sequence.",
            icon="/public/code.svg",
        ),
    ]
def extract_file_content(file_path: Path) -> Optional[str]:
    """
    Extract text content from an uploaded file for display/preview.

    Args:
        file_path: Path to the uploaded file.

    Returns:
        str: Extracted text content, or None if extraction failed.
    """
    suffix = file_path.suffix.lower()

    try:
        # Text-based files
        if suffix in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8")

        # PDF files
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(file_path)
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts) if text_parts else None
            except ImportError:
                return f"[PDF file uploaded: {file_path.name} - pypdf not installed for extraction]"

        else:
            return f"[File uploaded: {file_path.name}]"

    except Exception as e:
        return f"[Error reading file {file_path.name}: {str(e)}]"
