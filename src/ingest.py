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
CHROMA_DB_PATH: Path = Path("data/chromadb")
COLLECTION_NAME: str = "personal_knowledge"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast and efficient
CHUNK_SIZE: int = 500  # Characters per chunk
CHUNK_OVERLAP: int = 50  # Overlap between chunks for context continuity
DATA_DIRECTORY: Path = Path("data")
SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".pdf"}
SKIP_DIRECTORIES: set[str] = {"chromadb", "uploads", "__pycache__", ".git"}
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
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
