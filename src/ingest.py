#!/usr/bin/env python3
"""
Data Ingestion Layer for Infinite Memory LLM Agent.

This module handles the ingestion of documents from the local `data/` directory
into ChromaDB for vector-based retrieval. Documents are chunked, embedded using
sentence-transformers, and stored for RAG (Retrieval-Augmented Generation).

Supported formats: .txt, .md, .pdf

Usage:
    python -m src.ingest
    or
    python src/ingest.py
    
    # Ingest a specific file
    python -m src.ingest path/to/file.pdf
"""

import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# ChromaDB Configuration
CHROMA_DB_PATH: Path = Path("data/chromadb")
COLLECTION_NAME: str = "personal_knowledge"

# Embedding Model
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast and efficient

# Chunking Configuration
CHUNK_SIZE: int = 500  # Characters per chunk
CHUNK_OVERLAP: int = 50  # Overlap between chunks for context continuity

# Directory containing documents to ingest
DATA_DIRECTORY: Path = Path("data")

# Supported file extensions
SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".pdf"}

# Directories to skip during scanning
SKIP_DIRECTORIES: set[str] = {"chromadb", "uploads", "__pycache__", ".git"}

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------


class DocumentIngester:
    """
    Handles document ingestion into ChromaDB.
    
    Features:
    - PDF, TXT, MD file support
    - Text chunking with overlap
    - Embedding via sentence-transformers
    - Deduplication using content hashing
    """
    
    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the document ingester.
        
        Args:
            chroma_path: Path to ChromaDB storage directory.
            embedding_model: Name of the sentence-transformer model.
        """
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        
        # Initialize ChromaDB
        self.chroma_client = self._init_chromadb()
        self.collection = self._get_or_create_collection()
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("✓ Embedding model loaded")
    
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
        """Get or create the document collection."""
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Personal knowledge base for infinite memory"}
            )
            count = collection.count()
            logger.info(f"✓ Collection '{COLLECTION_NAME}' ready ({count} existing documents)")
            return collection
        except Exception as e:
            logger.error(f"✗ Failed to get/create collection: {e}")
            raise
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for content deduplication."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    
    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full text content to chunk.
        
        Returns:
            List of text chunks.
        """
        if len(text) <= CHUNK_SIZE:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end of chunk
            end = start + CHUNK_SIZE
            
            # Try to break at sentence/paragraph boundary
            if end < len(text):
                # Look for sentence endings
                for boundary in [". ", ".\n", "\n\n", "\n", " "]:
                    break_point = text.rfind(boundary, start + CHUNK_SIZE // 2, end + 50)
                    if break_point != -1:
                        end = break_point + len(boundary)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - CHUNK_OVERLAP
            if start < 0:
                start = end
        
        return chunks
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            Text content or None if reading failed.
        """
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in {".txt", ".md"}:
                return file_path.read_text(encoding="utf-8")
            
            elif suffix == ".pdf":
                return self._extract_pdf_text(file_path)
            
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return None
                
        except UnicodeDecodeError as e:
            logger.error(f"✗ Encoding error reading '{file_path.name}': {e}")
            return None
        except Exception as e:
            logger.error(f"✗ Failed to read '{file_path.name}': {e}")
            return None
    
    def _extract_pdf_text(self, file_path: Path) -> Optional[str]:
        """Extract text from a PDF file using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning(
                f"✗ Cannot read PDF '{file_path.name}': pypdf not installed. "
                "Install with: pip install pypdf"
            )
            return None
        
        try:
            reader = PdfReader(file_path)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    logger.debug(f"Page {page_num} of '{file_path.name}' yielded no text")
            
            if not text_parts:
                logger.warning(f"✗ No extractable text found in '{file_path.name}'")
                return None
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"✗ PDF extraction failed for '{file_path.name}': {e}")
            return None
    
    def ingest_text(
        self,
        content: str,
        source: str,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Ingest raw text content into ChromaDB.
        
        Args:
            content: Text content to ingest.
            source: Source identifier (e.g., filename).
            metadata: Additional metadata to store.
        
        Returns:
            Number of chunks inserted.
        """
        if not content or not content.strip():
            logger.warning("Empty content, skipping")
            return 0
        
        # Chunk the content
        chunks = self._chunk_text(content)
        
        if not chunks:
            logger.warning("No chunks generated, skipping")
            return 0
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            # Generate unique ID from content hash
            chunk_hash = self._generate_content_hash(chunk)
            doc_id = f"{source}_{chunk_hash[:8]}_{i}"
            
            # Check if already exists (deduplication)
            try:
                existing = self.collection.get(ids=[doc_id])
                if existing and existing["ids"]:
                    logger.debug(f"Chunk {i} already exists, skipping")
                    continue
            except Exception:
                pass  # ID doesn't exist, proceed
            
            # Generate embedding
            embedding = self.embedding_model.encode(chunk).tolist()
            
            # Prepare metadata
            chunk_metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content_hash": chunk_hash,
            }
            if metadata:
                chunk_metadata.update(metadata)
            
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)
            embeddings.append(embedding)
        
        if not ids:
            logger.info(f"All chunks from '{source}' already exist, skipping")
            return 0
        
        # Insert into ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.info(f"✓ Inserted {len(ids)} chunks from '{source}'")
            return len(ids)
            
        except Exception as e:
            logger.error(f"✗ Failed to insert chunks: {e}")
            return 0
    
    def ingest_file(self, file_path: Path) -> int:
        """
        Ingest a single file into ChromaDB.
        
        Args:
            file_path: Path to the file to ingest.
        
        Returns:
            Number of chunks inserted.
        """
        logger.info(f"  → Processing: {file_path.name}")
        
        # Read file content
        content = self._read_file(file_path)
        if not content:
            return 0
        
        # Ingest with file metadata
        metadata = {
            "file_path": str(file_path.absolute()),
            "file_type": file_path.suffix.lower(),
        }
        
        return self.ingest_text(
            content=content,
            source=file_path.name,
            metadata=metadata
        )
    
    def discover_files(self, data_dir: Path) -> list[Path]:
        """
        Scan directory for supported document files.
        
        Args:
            data_dir: Directory to scan.
        
        Returns:
            List of file paths to ingest.
        """
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory '{data_dir}' does not exist. "
                "Please create it and add documents."
            )
        
        if not data_dir.is_dir():
            raise NotADirectoryError(f"'{data_dir}' is not a directory.")
        
        discovered_files = []
        
        logger.info(f"Scanning '{data_dir}' for documents...")
        
        for file_path in data_dir.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue
            
            # Skip files in excluded directories
            if any(skip_dir in file_path.parts for skip_dir in SKIP_DIRECTORIES):
                continue
            
            # Skip hidden files
            if any(part.startswith(".") for part in file_path.parts):
                continue
            
            # Check extension
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                discovered_files.append(file_path)
        
        discovered_files.sort()
        
        logger.info(
            f"✓ Discovered {len(discovered_files)} file(s) with extensions: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}"
        )
        
        return discovered_files
    
    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        return {
            "chroma_path": str(self.chroma_path),
            "collection_name": COLLECTION_NAME,
            "document_count": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }


def run_ingestion(data_dir: Optional[Path] = None, single_file: Optional[Path] = None) -> dict:
    """
    Execute the complete ingestion pipeline.
    
    Args:
        data_dir: Directory to scan for files.
        single_file: Single file to ingest (overrides data_dir scan).
    
    Returns:
        Statistics dictionary.
    """
    logger.info("=" * 60)
    logger.info("DOCUMENT INGESTION - Starting")
    logger.info("=" * 60)
    
    stats = {"total": 0, "success": 0, "failed": 0, "chunks_added": 0}
    
    # Initialize ingester
    ingester = DocumentIngester()
    
    # Determine files to process
    if single_file:
        if not single_file.exists():
            raise FileNotFoundError(f"File '{single_file}' does not exist.")
        files = [single_file]
    else:
        files = ingester.discover_files(data_dir or DATA_DIRECTORY)
    
    if not files:
        logger.warning("No files found to ingest. Add documents to the 'data/' directory.")
        return stats
    
    stats["total"] = len(files)
    
    # Process files
    logger.info("-" * 60)
    logger.info("Beginning file ingestion...")
    logger.info("-" * 60)
    
    for file_path in files:
        try:
            chunks_added = ingester.ingest_file(file_path)
            if chunks_added >= 0:
                stats["success"] += 1
                stats["chunks_added"] += chunks_added
            else:
                stats["failed"] += 1
        except Exception as e:
            logger.error(f"✗ Failed to ingest '{file_path.name}': {e}")
            stats["failed"] += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Total files:    {stats['total']}")
    logger.info(f"  Successful:     {stats['success']}")
    logger.info(f"  Failed:         {stats['failed']}")
    logger.info(f"  Chunks added:   {stats['chunks_added']}")
    logger.info(f"  Total in DB:    {ingester.collection.count()}")
    logger.info("=" * 60)
    
    return stats


# -----------------------------------------------------------------------------
# Convenience Functions for External Use
# -----------------------------------------------------------------------------


def ingest_file(file_path: Path) -> int:
    """
    Convenience function to ingest a single file.
    
    Args:
        file_path: Path to the file to ingest.
    
    Returns:
        Number of chunks added.
    """
    ingester = DocumentIngester()
    return ingester.ingest_file(file_path)


def ingest_text(content: str, source: str = "direct_input") -> int:
    """
    Convenience function to ingest raw text.
    
    Args:
        content: Text content to ingest.
        source: Source identifier.
    
    Returns:
        Number of chunks added.
    """
    ingester = DocumentIngester()
    return ingester.ingest_text(content, source)


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------


def main() -> int:
    """Main entry point with argument handling."""
    try:
        # Check for single file argument
        single_file = None
        if len(sys.argv) > 1:
            single_file = Path(sys.argv[1])
        
        stats = run_ingestion(single_file=single_file)
        
        if stats["total"] == 0:
            return 0
        elif stats["failed"] == 0:
            return 0
        elif stats["success"] > 0:
            return 1
        else:
            return 2
            
    except FileNotFoundError as e:
        logger.critical(str(e))
        return 2
    except KeyboardInterrupt:
        logger.info("\nIngestion cancelled by user.")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error during ingestion: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
