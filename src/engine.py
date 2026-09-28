"""
Memory Agent Engine - Core Brain for Infinite Memory LLM Agent.

This module provides a lightweight, custom agent that replaces complex frameworks
with pure Python, ChromaDB for vector storage, and Ollama for LLM inference.

Key Features:
    - Core Memory: Persistent user info and persona (data/memory.json)
    - Archival Memory: Document embeddings in ChromaDB
    - RAG: Context retrieval from ChromaDB based on user queries
    - Tool Execution: JSON-based tool calling with automatic execution
    - Internal Monologue: Chain-of-thought reasoning before responses
    - Active Modes: Dynamic persona injection (Teacher Mode, Coding Guide, etc.)

Usage:
    from src.engine import MemoryAgent
    
    agent = MemoryAgent()
    agent.activate_mode("Teacher Mode", "You are a Socratic tutor...")
    response = agent.chat("Hello, who am I?")
    print(response["answer"])
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import chromadb
import ollama
OLLAMA_MODEL: str = "hermes3:8b"
CHROMA_DB_PATH: Path = Path("data/chromadb")
COLLECTION_NAME: str = "personal_knowledge"
MEMORY_FILE: Path = Path("data/memory.json")
HISTORY_FILE: Path = Path("data/history.json")
DEFAULT_CORE_MEMORY = {
    "user": {
        "name": "Unknown",
        "preferences": [],
        "notes": ""
    },
    "persona": {
        "name": "Jarvis",
        "traits": [
            "helpful",
            "precise",
            "proactive",
            "knowledgeable"
        ],
        "role": "AI assistant with infinite memory capabilities"
    },
    "conversation_context": {
        "topics_discussed": [],
        "last_interaction": None
    }
}
RAG_TOP_K: int = 3  # Number of documents to retrieve
RAG_MIN_SIMILARITY: float = 0.3  # Minimum similarity threshold
MAX_HISTORY_MESSAGES: int = 10  # Keep last N messages in context
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
class MemoryAgent:
    """
    A lightweight AI agent with persistent memory and tool capabilities.
    
    This agent uses:
    - ChromaDB for vector-based document retrieval (archival memory)
    - JSON file for core memory (user info, persona)
    - Ollama for LLM inference
    - Custom tool execution via JSON parsing
    - Active Modes for dynamic persona injection
    
    Attributes:
        core_memory: Dict containing user and persona information
        chroma_client: ChromaDB persistent client
        collection: ChromaDB collection for document embeddings
        conversation_history: List of recent messages for context
        active_modes: List of active persona/behavior modifiers
    """
    def __init__(self, memory_file: Optional[Path] = None, chroma_path: Optional[Path] = None):
        """
        Initialize the Memory Agent.
        
        Args:
            memory_file: Path to the core memory JSON file.
            chroma_path: Path to the ChromaDB storage directory.
        """
        self.memory_file = memory_file or MEMORY_FILE
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        
        # Initialize components
        self.core_memory = self._load_core_memory()
        self.chroma_client = self._init_chromadb()
        self.collection = self._get_or_create_collection()
        self.conversation_history: list[dict[str, str]] = []
        
        # Active modes for dynamic persona injection
        # Each mode is a dict: {"name": str, "instruction": str}
        self.active_modes: list[dict[str, str]] = []
        
        # Import tools lazily to avoid circular imports
        self._tools = None
        
        logger.info(f"✓ MemoryAgent initialized with model: {OLLAMA_MODEL}")
    def activate_mode(self, mode_name: str, instruction: str) -> None:
        """
        Activate a mode by injecting a persona instruction into the system prompt.
        
        This allows dynamic behavior changes like "Teacher Mode" or "Coding Guide"
        without modifying the core persona.
        
        Args:
            mode_name: A unique name for this mode (e.g., "Teacher Mode").
            instruction: The full instruction to inject into the system prompt.
        """
        # Remove any existing mode with the same name first
        self.deactivate_mode(mode_name)
        
        self.active_modes.append({
            "name": mode_name,
            "instruction": instruction
        })
        logger.info(f"✓ Activated mode: {mode_name}")
    def deactivate_mode(self, mode_name: str) -> None:
        """
        Deactivate a mode by name.
        
        Args:
            mode_name: The name of the mode to remove.
        """
        original_count = len(self.active_modes)
        self.active_modes = [m for m in self.active_modes if m["name"] != mode_name]
        
        if len(self.active_modes) < original_count:
            logger.info(f"✓ Deactivated mode: {mode_name}")
    def clear_all_modes(self) -> None:
        """Clear all active modes, returning to default behavior."""
        self.active_modes.clear()
        logger.info("✓ All modes cleared")
    def get_active_mode_names(self) -> list[str]:
        """Get the names of all currently active modes."""
        return [m["name"] for m in self.active_modes]
    def _load_core_memory(self) -> dict:
        """Load core memory from JSON file, creating default if not exists."""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    memory = json.load(f)
                logger.info(f"✓ Loaded core memory from {self.memory_file}")
                return memory
            else:
                logger.info("No existing memory file, creating default...")
                self._save_core_memory(DEFAULT_CORE_MEMORY)
                return DEFAULT_CORE_MEMORY.copy()
        except Exception as e:
            logger.error(f"✗ Failed to load core memory: {e}")
            return DEFAULT_CORE_MEMORY.copy()
    def _save_core_memory(self, memory: Optional[dict] = None) -> None:
        """Save core memory to JSON file."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(memory or self.core_memory, f, indent=2, ensure_ascii=False)
            logger.debug(f"✓ Saved core memory to {self.memory_file}")
        except Exception as e:
            logger.error(f"✗ Failed to save core memory: {e}")
    def update_core_memory(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific field in core memory.
        
        Args:
            section: Top-level section (e.g., 'user', 'persona')
            key: Key within the section
            value: New value to set
        """
        if section in self.core_memory:
            self.core_memory[section][key] = value
            self._save_core_memory()
            logger.info(f"Updated core memory: {section}.{key}")
    def get_core_memory_string(self) -> str:
        """Format core memory as a readable string for the system prompt."""
        user = self.core_memory.get("user", {})
        persona = self.core_memory.get("persona", {})
        
        lines = [
            "=== CORE MEMORY ===",
            f"User Name: {user.get('name', 'Unknown')}",
            f"User Notes: {user.get('notes', 'None')}",
            f"User Preferences: {', '.join(user.get('preferences', [])) or 'None'}",
            "",
            f"My Name: {persona.get('name', 'Jarvis')}",
            f"My Role: {persona.get('role', 'AI Assistant')}",
            f"My Traits: {', '.join(persona.get('traits', []))}",
        ]
        return "\n".join(lines)
    def _load_history(self) -> list[dict]:
        """
        Load recent conversation history from JSONL.
        Efficiently reads only the last N lines.
        """
        if self.memory_file.parent.joinpath("history.jsonl").exists():
            history = []
            try:
                with open(self.memory_file.parent / "history.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            history.append(json.loads(line))
                
                # Return only the last N messages
                return history[-MAX_HISTORY_MESSAGES:]
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
        return []
    def _append_to_history(self, user_msg: str, ai_msg: str) -> None:
        """
        Append the new interaction to the history file as a single line (JSONL).
        This is O(1) - instant, regardless of file size.
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Prepare entries
            user_entry = {"role": "user", "content": user_msg, "timestamp": timestamp}
            ai_entry = {"role": "assistant", "content": ai_msg, "timestamp": timestamp}
            
            # Append directly to file
            with open(self.memory_file.parent / "history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(user_entry) + "\n")
                f.write(json.dumps(ai_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    def _archive_interaction(self, user_msg: str, ai_msg: str) -> None:
        """Save the interaction to ChromaDB for long-term recall."""
        try:
            import uuid
            from datetime import datetime
            
            # Create a searchable text blob
            content = f"Date: {datetime.now().isoformat()}\nUser: {user_msg}\nJarvis: {ai_msg}"
            
            self.collection.add(
                documents=[content],
                metadatas=[{"source": "conversation_history", "date": datetime.now().isoformat()}],
                ids=[str(uuid.uuid4())]
            )
            logger.debug("✓ Archived interaction to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to archive interaction: {e}")
    def _init_chromadb(self) -> chromadb.PersistentClient:
        """Initialize ChromaDB persistent client."""
        try:
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            logger.info(f"✓ ChromaDB initialized at {self.chroma_path}")
            return client
        except Exception as e:
            logger.error(f"✗ Failed to initialize ChromaDB: {e}")
            raise
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get or create the document collection in ChromaDB."""
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Personal knowledge base for infinite memory"}
            )
            count = collection.count()
            logger.info(f"✓ Collection '{COLLECTION_NAME}' ready ({count} documents)")
            return collection
        except Exception as e:
            logger.error(f"✗ Failed to get/create collection: {e}")
            raise
    def query_archival_memory(self, query: str, top_k: int = RAG_TOP_K) -> list[str]:
        """
        Query ChromaDB for relevant document chunks.
        
        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
        
        Returns:
            List of relevant document excerpts.
        """
        try:
            # Check if collection is empty
            if self.collection.count() == 0:
                logger.debug("Archival memory is empty, skipping RAG")
                return []
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
            )
            
            # Extract documents with metadata
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            # Filter by similarity and format
            relevant_docs = []
            for doc, dist, meta in zip(documents, distances, metadatas):
                # ChromaDB uses L2 distance; lower is more similar
                # Convert to similarity score (approximate)
                similarity = 1 / (1 + dist)
                
                if similarity >= RAG_MIN_SIMILARITY:
                    source = meta.get("source", "Unknown")
                    relevant_docs.append(f"[Source: {source}]\n{doc}")
            
            logger.debug(f"RAG: Found {len(relevant_docs)} relevant documents")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"✗ RAG query failed: {e}")
            return []
    def _get_tools(self) -> dict:
        """Lazy-load tools to avoid circular imports."""
        if self._tools is None:
            AVAILABLE_TOOLS = None
            run_tool = None
            
            # Try multiple import paths for robustness
            try:
                from src.tools import AVAILABLE_TOOLS, run_tool
                logger.debug("Imported tools from src.tools")
            except ImportError:
                try:
                    from tools import AVAILABLE_TOOLS, run_tool
                    logger.debug("Imported tools from tools")
                except ImportError:
                    logger.warning("Could not import tools module from any path")
                    self._tools = {"registry": {}, "runner": lambda n, a: "Tool not found"}
                    return self._tools

            self._tools = {
                "registry": AVAILABLE_TOOLS,
                "runner": run_tool
            }
        return self._tools
    def get_tools_description(self) -> str:
        """Get a formatted description of available tools for the system prompt."""
        tools = self._get_tools()["registry"]
        
        if not tools:
            return "No tools available."
        
        lines = ["=== AVAILABLE TOOLS ==="]
        for name, info in tools.items():
            desc = info.get("description", "No description")
            params = info.get("parameters", [])
            params_str = ", ".join(params) if params else "none"
            lines.append(f"\n• {name}({params_str})")
            lines.append(f"  Description: {desc}")
        
        lines.append("\n\nTo use a tool, respond with ONLY a JSON object:")
        lines.append('{"tool": "tool_name", "args": {"param1": "value1"}}')
        
        return "\n".join(lines)
