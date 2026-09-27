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
