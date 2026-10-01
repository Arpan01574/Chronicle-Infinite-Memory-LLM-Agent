#!/usr/bin/env python3
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

# =============================================================================
# CRITICAL: Fix imports BEFORE anything else
# This ensures 'engine', 'tools', and 'ingest' can always be found
# regardless of how the script is invoked.
# =============================================================================
_src_dir = Path(__file__).parent.absolute()
_project_root = _src_dir.parent
os.chdir(_project_root)

_venv_chainlit = os.path.join(_project_root, ".venv", "Scripts", "chainlit.exe")
if os.path.exists(_venv_chainlit) and "chainlit" not in os.path.basename(sys.argv[0]).lower():
    import socket
    import subprocess
    import webbrowser

    def _is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(('127.0.0.1', port)) == 0

    target_port = 8000
    if _is_port_in_use(target_port):
        print(f"\n[OK] Chronicle app is ALREADY running on http://localhost:{target_port}!")
        print(f"[OK] Opening http://localhost:{target_port} in your browser...\n")
        webbrowser.open(f"http://localhost:{target_port}")
        sys.exit(0)

    sys.exit(subprocess.call([_venv_chainlit, "run", str(Path(__file__).absolute()), "--port", str(target_port)] + sys.argv[1:], cwd=str(_project_root)))

if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# =============================================================================
# Now safe to import everything else
# =============================================================================
import shutil
from datetime import datetime
from typing import Optional

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select

# Import from our modules (now works thanks to path fix)
from engine import MemoryAgent
from ingest import ingest_file
from tools import fetch_coding_guidelines

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# File upload directory
UPLOADS_DIR: Path = Path("data/uploads")

# Database path for chat persistence
DATABASE_PATH: Path = Path("data/chainlit.db")

# Supported file types for upload
SUPPORTED_UPLOAD_TYPES = {".pdf", ".txt", ".md"}

# Mode prompts - these get injected into the agent's system prompt
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


# -----------------------------------------------------------------------------
# Data Layer Configuration (Chat Persistence)
# -----------------------------------------------------------------------------


@cl.data_layer
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


# -----------------------------------------------------------------------------
# Author Rename - Gives the assistant a personality
# -----------------------------------------------------------------------------


@cl.author_rename
def rename_author(orig_author: str) -> str:
    """Rename the default author names for a more personalized experience."""
    rename_map = {
        "Assistant": "Jarvis",
        "assistant": "Jarvis",
        "Chatbot": "Jarvis",
    }
    return rename_map.get(orig_author, orig_author)


# -----------------------------------------------------------------------------
# Chat Settings - Mode Selection (Replaces Toolbox)
# -----------------------------------------------------------------------------


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

@cl.on_settings_update
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


# -----------------------------------------------------------------------------
# Chat Starters - Gemini-style quick action buttons
# -----------------------------------------------------------------------------


@cl.set_starters
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


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Chat Lifecycle Hooks
# -----------------------------------------------------------------------------


@cl.on_chat_start
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


@cl.on_chat_resume
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


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages.

    Processes:
    - Text messages sent to the Memory Agent
    - File uploads with automatic ingestion
    - Displays inner monologue and tool calls
    """
    # Retrieve session state
    agent: Optional[MemoryAgent] = cl.user_session.get("agent")

    if not agent:
        await cl.Message(
            content="❌ Session not initialized. Please refresh the page.",
            author="System"
        ).send()
        return

    # Prepare the input message
    input_message = message.content

    # Handle file uploads
    uploaded_files_info = []
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                # Check if file type is supported
                file_ext = Path(element.name).suffix.lower()
                if file_ext not in SUPPORTED_UPLOAD_TYPES:
                    await cl.Message(
                        content=f"⚠️ Unsupported file type: `{file_ext}`. Supported: {', '.join(SUPPORTED_UPLOAD_TYPES)}",
                        author="System"
                    ).send()
                    continue

                # Save the uploaded file
                saved_path = await save_uploaded_file(element)
                if saved_path:
                    # Ingest file into ChromaDB
                    await cl.Message(
                        content=f"📎 Processing file: **{element.name}**...",
                        author="System"
                    ).send()
                    
                    try:
                        chunks_added = ingest_file(saved_path)
                        await cl.Message(
                            content=f"✅ Ingested **{element.name}** ({chunks_added} chunks added to knowledge base)",
                            author="System"
                        ).send()
                    except Exception as e:
                        await cl.Message(
                            content=f"⚠️ Could not ingest file: {str(e)}",
                            author="System"
                        ).send()
                    
                    # Extract content for context
                    content = extract_file_content(saved_path)
                    preview = content[:1000] + "..." if content and len(content) > 1000 else content

                    uploaded_files_info.append({
                        "name": element.name,
                        "path": str(saved_path.absolute()),
                        "preview": preview,
                    })

    # Append file context to the message if files were uploaded
    if uploaded_files_info:
        file_context = "\n\n[SYSTEM: User uploaded files that have been added to memory]\n"
        for file_info in uploaded_files_info:
            file_context += f"\n📄 FILE: {file_info['name']}\n"
            file_context += f"   Preview: {file_info['preview'][:500]}...\n"

        input_message = (
            input_message + file_context 
            if input_message 
            else f"Please analyze and summarize the file I just uploaded: {file_info['name']}"
        )

    # Show thinking indicator
    thinking_msg = cl.Message(content="🤔 Thinking...")
    await thinking_msg.send()

    try:
        # Send message to Memory Agent
        response = agent.chat(input_message)

        # Remove thinking indicator
        await thinking_msg.remove()

        # Display active modes if any
        if response.get("active_modes"):
            modes_str = ", ".join(response["active_modes"])
            async with cl.Step(name=f"🎭 Active Mode: {modes_str}", type="run") as step:
                step.output = f"Responding in {modes_str} mode"

        # Display RAG context if any
        if response.get("rag_documents"):
            async with cl.Step(name="📚 Retrieved Knowledge", type="retrieval") as step:
                docs_text = "\n\n---\n\n".join(response["rag_documents"][:3])
                step.output = docs_text[:2000] + "..." if len(docs_text) > 2000 else docs_text

        # Display inner monologue in a collapsible step
        if response.get("thoughts"):
            async with cl.Step(name="💭 Inner Monologue", type="llm") as step:
                step.output = "\n\n".join(response["thoughts"])

        # Display tool calls if any
        if response.get("tool_calls"):
            async with cl.Step(name="🔧 Tool Calls", type="tool") as step:
                tool_lines = []
                for tc in response["tool_calls"]:
                    tool_lines.append(f"**{tc['tool']}**\nArgs: `{tc['args']}`")
                step.output = "\n\n".join(tool_lines)

        # Send the final response
        final_answer = response.get("answer", "")
        if final_answer:
            await cl.Message(content=final_answer).send()
        else:
            await cl.Message(
                content="I processed your request but have no direct response. Check the inner monologue for details.",
            ).send()

    except Exception as e:
        # Remove thinking indicator on error
        await thinking_msg.remove()

        await cl.Message(
            content=(
                f"❌ **Error processing your request**\n\n"
                f"```\n{str(e)}\n```\n\n"
                f"Please check:\n"
                f"1. Is Ollama running? (`ollama serve`)\n"
                f"2. Is the model pulled? (`ollama pull hermes3:8b`)\n"
            ),
            author="System"
        ).send()


@cl.on_chat_end
async def on_chat_end():
    """Called when the chat session ends. Cleanup any resources if needed."""
    print("Chat session ended")


@cl.on_stop
async def on_stop():
    """Called when the user stops a running response."""
    agent: Optional[MemoryAgent] = cl.user_session.get("agent")
    if agent:
        print("User stopped the response")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Run this application with: chainlit run src/app.py -w")
    print("The -w flag enables hot reloading during development.")
