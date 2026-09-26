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
