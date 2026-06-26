"""Document parsing and chunking module for GraphRAG engine.

This module handles document ingestion, preprocessing, and intelligent chunking
to prepare text for entity extraction and graph construction.
"""

import re
import uuid
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Iterator, Tuple
from dataclasses import dataclass
import tiktoken

from ..schemas import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_size: int = 100
    separator: str = "\n\n"
    
    def __post_init__(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")


class DocumentParser:
    """Parses and chunks documents for the GraphRAG pipeline.
    
    Supports markdown files with intelligent section detection and
    token-aware chunking to optimize LLM processing.
    """
    
    # Markdown section patterns for intelligent parsing
    MARKDOWN_HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize the document parser.
        
        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkConfig()
        self._encoding = None
    
    def _get_encoding(self):
        """Lazy-load tiktoken encoding for token counting."""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.warning("Failed to load tiktoken, falling back to approximate token count")
                self._encoding = None
        return self._encoding
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        encoding = self._get_encoding()
        if encoding:
            return len(encoding.encode(text))
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with normalized whitespace and formatting
        """
        # Remove excessive whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Preserve code blocks but normalize internal formatting
        def clean_code(match):
            code = match.group(0)
            code = re.sub(r'\n{2,}', '\n', code)
            return code
        
        text = self.CODE_BLOCK_PATTERN.sub(clean_code, text)
        
        return text.strip()
    
    def _split_by_headings(self, text: str) -> List[Tuple[str, str]]:
        """Split markdown text into sections by headings.
        
        Args:
            text: Markdown text to split
            
        Returns:
            List of (heading, content) tuples
        """
        sections = []
        current_heading = ""
        current_content = []
        
        lines = text.split('\n')
        for line in lines:
            heading_match = self.MARKDOWN_HEADING_PATTERN.match(line)
            if heading_match:
                # Save previous section
                if current_content:
                    sections.append((current_heading, '\n'.join(current_content)))
                    current_content = []
                current_heading = heading_match.group(2).strip()
            current_content.append(line)
        
        # Add final section
        if current_content:
            sections.append((current_heading, '\n'.join(current_content)))
        
        return sections
    
    def _chunk_text(self, text: str, doc_name: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks with token awareness.
        
        Args:
            text: Text to chunk
            doc_name: Document name for metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        separators = [self.config.separator, "\n", ". ", " "]
        
        # First try semantic splitting by headings
        sections = self._split_by_headings(text)
        
        if len(sections) > 1:
            # Process each section separately
            for heading, content in sections:
                if not content.strip():
                    continue
                    
                section_chunks = self._chunk_by_tokens(content, doc_name, heading)
                chunks.extend(section_chunks)
        else:
            # Fall back to token-based chunking
            chunks = self._chunk_by_tokens(text, doc_name, "")
        
        return chunks
    
    def _chunk_by_tokens(self, text: str, doc_name: str, section: str) -> List[DocumentChunk]:
        """Chunk text based on token count.
        
        Args:
            text: Text to chunk
            doc_name: Document name
            section: Section heading if applicable
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        encoding = self._get_encoding()
        
        # If we have encoding, use token-based chunking
        if encoding:
            tokens = encoding.encode(text)
            token_count = len(tokens)
            
            if token_count <= self.config.chunk_size:
                chunk_id = self._generate_chunk_id(doc_name, 0)
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    content=text,
                    document_name=doc_name,
                    chunk_index=0,
                    total_chunks=1,
                    metadata={"section": section} if section else {}
                ))
                return chunks
            
            # Sliding window chunking
            start = 0
            chunk_idx = 0
            while start < token_count:
                end = min(start + self.config.chunk_size, token_count)
                
                # Try to break at a separator
                if end < token_count:
                    for sep in separators:
                        sep_token = encoding.encode(sep)[0]
                        for i in range(end - 1, max(start, end - 100), -1):
                            if tokens[i] == sep_token:
                                end = i + 1
                                break
                        else:
                            continue
                        break
                
                chunk_text = encoding.decode(tokens[start:end])
                chunk_id = self._generate_chunk_id(doc_name, chunk_idx)
                
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    content=chunk_text.strip(),
                    document_name=doc_name,
                    chunk_index=chunk_idx,
                    total_chunks=-1,  # Will be updated
                    metadata={"section": section} if section else {}
                ))
                
                chunk_idx += 1
                start = end - self.config.overlap if end < token_count else end
        else:
            # Fallback: character-based chunking
            chunk_size_chars = self.config.chunk_size * 4  # Approximate
            overlap_chars = self.config.overlap * 4
            
            for i in range(0, len(text), chunk_size_chars - overlap_chars):
                chunk_text = text[i:i + chunk_size_chars]
                if len(chunk_text) < self.config.min_chunk_size and chunks:
                    break
                    
                chunk_id = self._generate_chunk_id(doc_name, len(chunks))
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    content=chunk_text.strip(),
                    document_name=doc_name,
                    chunk_index=len(chunks),
                    total_chunks=-1,
                    metadata={"section": section} if section else {}
                ))
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _generate_chunk_id(self, doc_name: str, index: int) -> str:
        """Generate a unique chunk ID.
        
        Args:
            doc_name: Document name
            index: Chunk index
            
        Returns:
            Unique chunk identifier
        """
        content = f"{doc_name}:{index}"
        hash_digest = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"chunk_{hash_digest}_{index}"
    
    def parse_file(self, file_path: Path) -> List[DocumentChunk]:
        """Parse a markdown file and return chunks.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            List of DocumentChunk objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not readable or empty
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='latin-1')
        
        if not content.strip():
            raise ValueError(f"File is empty: {file_path}")
        
        doc_name = file_path.stem
        cleaned = self._clean_text(content)
        chunks = self._chunk_text(cleaned, doc_name)
        
        logger.info(f"Parsed {file_path.name}: {len(chunks)} chunks created")
        return chunks
    
    def parse_text(self, text: str, doc_name: str = "text") -> List[DocumentChunk]:
        """Parse raw text and return chunks.
        
        Args:
            text: Raw text content
            doc_name: Document name identifier
            
        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            raise ValueError("Text content is empty")
        
        cleaned = self._clean_text(text)
        chunks = self._chunk_text(cleaned, doc_name)
        
        logger.info(f"Parsed text '{doc_name}': {len(chunks)} chunks created")
        return chunks
    
    def parse_directory(self, directory: Path, pattern: str = "*.md") -> Iterator[DocumentChunk]:
        """Parse all markdown files in a directory.
        
        Args:
            directory: Directory containing markdown files
            pattern: Glob pattern for file matching
            
        Yields:
            DocumentChunk objects from all matching files
        """
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Directory not found: {directory}")
            return
        
        for file_path in sorted(directory.glob(pattern)):
            try:
                chunks = self.parse_file(file_path)
                yield from chunks
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                continue
    
    def stream_chunks(self, chunks: List[DocumentChunk]) -> Iterator[DocumentChunk]:
        """Stream chunks one at a time.
        
        Args:
            chunks: List of chunks to stream
            
        Yields:
            DocumentChunk objects
        """
        for chunk in chunks:
            yield chunk
    
    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """Get statistics about the chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary of statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_tokens_estimate": 0
            }
        
        sizes = [len(c.content) for c in chunks]
        tokens = [self._count_tokens(c.content) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_tokens_estimate": sum(tokens),
            "avg_tokens_per_chunk": sum(tokens) / len(tokens)
        }
