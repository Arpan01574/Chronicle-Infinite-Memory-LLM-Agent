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
async def save_uploaded_file(element) -> Optional[Path]:
    """
    Save an uploaded file to the uploads directory.

    Args:
        element: The Chainlit file element.

    Returns:
        Path: The saved file path, or None if saving failed.
    """
    try:
        # Ensure uploads directory exists
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(element.name)
        new_filename = f"{timestamp}_{original_name.stem}{original_name.suffix}"
        dest_path = UPLOADS_DIR / new_filename

        # Copy file to uploads directory
        shutil.copy(element.path, dest_path)

        return dest_path

    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return None
async def on_chat_start():
    """
    Initialize the chat session.

    Called when a new chat session begins. Sets up:
    - MemoryAgent initialization
    - Session state storage
    """
    # Show a loading message
    init_msg = cl.Message(
        content="🔄 Initializing Jarvis...",
        author="System"
    )
    await init_msg.send()

    try:
        # Initialize the Memory Agent
        agent = MemoryAgent()
        
        # Store in session for later use
        cl.user_session.set("agent", agent)

        settings = await cl.ChatSettings(
            [
                Select(
                    id="agent_mode",
                    label="🎭 Agent Mode",
                    description="Select how Jarvis should behave",
                    values=list(MODE_PROMPTS.keys()),
                    initial_value="General Assistant",
                ),
            ]
        ).send()

        # Get memory stats for display
        stats = agent.get_memory_stats()
        docs_count = stats.get("documents_count", 0)

        # Remove loading message
        await init_msg.remove()

        # Welcome message
        await cl.Message(
            content=(
                "👋 Hello! I'm **Jarvis**, your AI assistant with infinite memory.\n\n"
                f"📊 **Memory Status**: {docs_count} documents in knowledge base\n\n"
                "**Quick Start:**\n"
                "- Use the **⚙️ Settings** (gear icon) to change my mode:\n"
                "  - 🎓 **Teacher Mode** - I'll teach via Socratic questioning\n"
                "  - 💻 **Python/JS Coder** - I'll follow strict coding standards\n"
                "  - 🔍 **Code Reviewer** - I'll review your code thoroughly\n\n"
                "- Try the **starter buttons** above for common tasks\n"
                "- **Upload files** (PDF, TXT, MD) for me to analyze\n\n"
                "How can I help you today?"
            ),
        ).send()

    except Exception as e:
        await init_msg.remove()
        await cl.Message(
            content=(
                f"❌ **Initialization Error**\n\n"
                f"Could not initialize the Memory Agent.\n\n"
                f"Please ensure:\n"
                f"1. Ollama is running with the `hermes3:8b` model\n"
                f"2. ChromaDB dependencies are installed\n\n"
                f"```bash\n"
                f"ollama pull hermes3:8b\n"
                f"pip install chromadb sentence-transformers ollama\n"
                f"```\n\n"
                f"Error: {str(e)}"
            ),
            author="System"
        ).send()
async def on_chat_resume(thread):
    """
    Resume a previous chat session.
    
    Called when a user selects a chat from the history sidebar.
    Reinitializes the MemoryAgent for the resumed session.
    """
    try:
        # Initialize the Memory Agent for this resumed session
        agent = MemoryAgent()
        cl.user_session.set("agent", agent)
        
        await cl.Message(
            content=(
                "🔄 **Session Resumed**\n\n"
                "Welcome back! I've loaded your previous conversation.\n"
                "Use **⚙️ Settings** to change my mode, or continue chatting!"
            ),
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"⚠️ Could not fully restore session: {str(e)}",
            author="System"
        ).send()
