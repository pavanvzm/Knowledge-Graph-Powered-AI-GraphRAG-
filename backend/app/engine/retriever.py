"""Hybrid Graph Retriever for GraphRAG Engine.

This module orchestrates the hybrid retrieval pipeline:
1. Vector search to find anchor entities
2. Graph traversal for multi-hop context
3. LLM synthesis for final answer generation
"""

import time
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from ..schemas import (
    GraphNode, GraphEdge, QueryRequest, QueryResponse,
    RetrievedContext, GraphVisualizationData, GraphVisualizationNode,
    GraphVisualizationEdge
)
from ..config import settings
from .graph_store import GraphStore
from .parser import DocumentParser

logger = logging.getLogger(__name__)


# =============================================================================
# Vector Index (Simple In-Memory Implementation)
# =============================================================================

class VectorIndex:
    """Simple in-memory vector index for semantic search.
    
    Uses cosine similarity for vector matching. In production,
    this should be replaced with FAISS, Chroma, or Pinecone.
    """
    
    def __init__(self, embedding_dimension: int = 1536):
        """Initialize the vector index.
        
        Args:
            embedding_dimension: Dimension of embedding vectors
        """
        self.embedding_dimension = embedding_dimension
        self._vectors: Dict[str, np.ndarray] = {}
        self._texts: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_vector(
        self, 
        chunk_id: str, 
        vector: np.ndarray, 
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a vector to the index.
        
        Args:
            chunk_id: Unique identifier for the chunk
            vector: Embedding vector
            text: Original text content
            metadata: Additional metadata
        """
        if vector.shape[0] != self.embedding_dimension:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} does not match "
                f"expected dimension {self.embedding_dimension}"
            )
        
        self._vectors[chunk_id] = vector / np.linalg.norm(vector)  # Normalize
        self._texts[chunk_id] = text
        self._metadata[chunk_id] = metadata or {}
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """Search for most similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score, text) tuples
        """
        if not self._vectors:
            return []
        
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # Compute cosine similarity with all vectors
        similarities = []
        for chunk_id, vector in self._vectors.items():
            score = np.dot(query_norm, vector)  # Cosine similarity (normalized)
            similarities.append((chunk_id, float(score), self._texts[chunk_id]))
        
        # Sort by score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_chunk_ids_by_entity_name(self, name: str) -> List[str]:
        """Find chunks containing a specific entity name.
        
        Args:
            name: Entity name to search for
            
        Returns:
            List of matching chunk IDs
        """
        results = []
        name_lower = name.lower()
        
        for chunk_id, text in self._texts.items():
            if name_lower in text.lower():
                results.append(chunk_id)
        
        return results
    
    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Get the text content of a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Text content or None
        """
        return self._texts.get(chunk_id)
    
    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Metadata dict or None
        """
        return self._metadata.get(chunk_id)
    
    def clear(self):
        """Clear the index."""
        self._vectors.clear()
        self._texts.clear()
        self._metadata.clear()
    
    def __len__(self) -> int:
        """Get number of indexed vectors."""
        return len(self._vectors)
    
    def get_all_texts(self) -> Dict[str, str]:
        """Get all indexed texts."""
        return self._texts.copy()


# =============================================================================
# Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """Generate embeddings using OpenAI or local models."""
    
    def __init__(self):
        """Initialize the embedding generator."""
        self._client = None
        self._model = settings.openai_embedding_model
        self._dimension = settings.embedding_dimension
    
    @property
    def client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.get_openai_api_key())
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self._client = None
        return self._client
    
    def generate(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None
        """
        if self.client is None:
            # Fallback: simple hash-based pseudo-embedding
            return self._generate_pseudo_embedding(text)
        
        try:
            response = self.client.embeddings.create(
                model=self._model,
                input=text[:8000]  # Token limit
            )
            embedding = response.data[0].embedding
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._generate_pseudo_embedding(text)
    
    def _generate_pseudo_embedding(self, text: str) -> np.ndarray:
        """Generate a simple pseudo-embedding from text hash.
        
        Used as fallback when OpenAI is unavailable.
        
        Args:
            text: Text to embed
            
        Returns:
            Pseudo-embedding vector
        """
        import hashlib
        
        # Create deterministic pseudo-embedding
        hash_input = text.encode()[:512]
        hash_bytes = hashlib.sha256(hash_input).digest()
        
        # Convert to numpy array with proper dimension
        vector = np.zeros(self._dimension, dtype=np.float32)
        for i, byte in enumerate(hash_bytes):
            idx = (i * 7) % self._dimension  # Spread bytes
            vector[idx] = (byte / 255.0) * 2 - 1  # Normalize to [-1, 1]
        
        return vector / np.linalg.norm(vector)  # Normalize


# =============================================================================
# LLM Synthesizer
# =============================================================================

class LLMSynthesizer:
    """Generate synthesized answers using LLM with graph context."""
    
    SYSTEM_PROMPT = """You are an expert knowledge graph assistant. Your task is to answer user questions based on the provided context from a knowledge graph.

The context contains structured information about entities (nodes) and their relationships (edges) extracted from documents. Use this structured information to provide accurate, comprehensive answers.

Guidelines:
1. Answer based primarily on the provided graph context
2. If the context doesn't contain enough information, say so clearly
3. Reference specific entities and relationships from the context
4. Explain your reasoning when helpful
5. Be concise but thorough
6. Format your answer with clear structure when appropriate
"""

    SYNTHESIS_TEMPLATE = """CONTEXT FROM KNOWLEDGE GRAPH:

Extracted Entities (Nodes):
{entities_text}

Relationships (Edges):
{edges_text}

Structured Facts:
{facts_text}

---

USER QUESTION: {question}

Provide a comprehensive answer based on the above context."""
    
    def __init__(self):
        """Initialize the LLM synthesizer."""
        self._client = None
    
    @property
    def client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.get_openai_api_key())
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self._client = None
        return self._client
    
    def synthesize(
        self,
        question: str,
        context: RetrievedContext,
        include_reasoning: bool = True
    ) -> Tuple[str, List[str], float]:
        """Synthesize an answer from the graph context.
        
        Args:
            question: User's question
            context: Retrieved graph context
            include_reasoning: Whether to include reasoning steps
            
        Returns:
            Tuple of (answer, reasoning_steps, confidence)
        """
        # Format entities
        entities_text = "\n".join([
            f"- {node.type}: {node.name}"
            + (f" - {node.properties.description}" if node.properties.description else "")
            for node in context.nodes
        ]) if context.nodes else "No entities found"
        
        # Format edges
        edges_text = "\n".join([
            f"- {edge.source_id} --[{edge.relationship_type}]--> {edge.target_id}"
            + (f": {edge.description}" if edge.description else "")
            for edge in context.edges
        ]) if context.edges else "No relationships found"
        
        # Generate facts from graph traversal
        facts_text = "\n".join(context.text_blocks) if context.text_blocks else "No structured facts available"
        
        prompt = self.SYNTHESIS_TEMPLATE.format(
            entities_text=entities_text,
            edges_text=edges_text,
            facts_text=facts_text,
            question=question
        )
        
        reasoning_steps = []
        confidence = 0.5  # Default confidence
        
        if self.client is None:
            # Fallback: generate simple answer without LLM
            answer = self._generate_fallback_answer(question, context)
            confidence = 0.3
        else:
            try:
                response = self.client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                answer = response.choices[0].message.content
                
                # Estimate confidence based on context relevance
                if len(context.nodes) > 0 and len(context.edges) > 0:
                    confidence = 0.7 + (min(len(context.nodes), 10) * 0.02)
                confidence = min(confidence, 0.95)
                
            except Exception as e:
                logger.error(f"LLM synthesis failed: {e}")
                answer = self._generate_fallback_answer(question, context)
                confidence = 0.2
        
        if include_reasoning:
            reasoning_steps.append(f"Found {len(context.nodes)} relevant entities")
            reasoning_steps.append(f"Traversed {len(context.edges)} relationships")
            if context.anchor_entities:
                reasoning_steps.append(f"Anchor entities: {', '.join(context.anchor_entities[:5])}")
        
        return answer, reasoning_steps, confidence
    
    def _generate_fallback_answer(
        self, 
        question: str, 
        context: RetrievedContext
    ) -> str:
        """Generate a simple fallback answer without LLM.
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Simple text answer
        """
        if not context.nodes:
            return "I couldn't find relevant information in the knowledge graph to answer your question."
        
        # Build a simple answer from entities
        node_names = [n.name for n in context.nodes[:5]]
        
        answer = "Based on the knowledge graph, I found the following relevant information:\n\n"
        answer += "**Entities:**\n" + "\n".join(f"- {name}" for name in node_names) + "\n\n"
        
        if context.text_blocks:
            answer += "**Key Information:**\n" + "\n".join(f"- {block[:200]}" for block in context.text_blocks[:3])
        else:
            answer += "The graph contains additional relationships that could provide more detailed answers."
        
        return answer


# =============================================================================
# Graph Context Formatter
# =============================================================================

class GraphContextFormatter:
    """Format graph traversal results into readable text blocks."""
    
    def format_subgraph(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> List[str]:
        """Format a subgraph into human-readable text blocks.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            List of formatted text blocks
        """
        text_blocks = []
        
        # Build node lookup
        node_map: Dict[str, GraphNode] = {n.id: n for n in nodes}
        
        # Format each edge as a statement
        for edge in edges:
            source = node_map.get(edge.source_id)
            target = node_map.get(edge.target_id)
            
            if not source or not target:
                continue
            
            # Create descriptive statement
            statements = []
            
            # Entity descriptions
            if source.properties.description:
                statements.append(f"{source.name} ({source.type}): {source.properties.description}")
            if target.properties.description:
                statements.append(f"{target.name} ({target.type}): {target.properties.description}")
            
            # Relationship statement
            rel_stmt = f"{source.name} {self._format_relationship(edge.relationship_type)} {target.name}"
            if edge.description:
                rel_stmt += f" - {edge.description}"
            statements.append(rel_stmt)
            
            text_blocks.append(". ".join(statements))
        
        # Group by relationship type
        by_type: Dict[str, List[str]] = defaultdict(list)
        for edge in edges:
            by_type[edge.relationship_type].append(edge.relationship_type)
        
        # Add summary blocks
        if by_type:
            summary = "Relationships found: " + ", ".join(
                f"{rel_type} ({count})" for rel_type, count in by_type.items()
            )
            text_blocks.insert(0, summary)
        
        return text_blocks
    
    def _format_relationship(self, rel_type: str) -> str:
        """Format relationship type to human-readable string.
        
        Args:
            rel_type: Relationship type constant
            
        Returns:
            Human-readable relationship phrase
        """
        relationships = {
            "DEPENDS_ON": "depends on",
            "OWNS": "is owned by",
            "USES": "uses",
            "CONNECTS_TO": "is connected to",
            "SUPPORTS": "supports",
            "IMPLEMENTS": "implements",
            "CONTAINS": "contains",
            "AUTHORED_BY": "was authored by",
            "REFERENCES": "references",
        }
        
        return relationships.get(rel_type, f"is related to ({rel_type})")


# =============================================================================
# Hybrid Graph Retriever
# =============================================================================

class HybridGraphRetriever:
    """Hybrid retrieval system combining vector search and graph traversal.
    
    This is the main orchestrator for the GraphRAG retrieval pipeline:
    1. Vector search to find anchor entities/chunks
    2. Graph traversal for multi-hop context
    3. Context formatting and LLM synthesis
    """
    
    def __init__(
        self,
        graph_store: GraphStore,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """Initialize the hybrid retriever.
        
        Args:
            graph_store: Graph store instance
            embedding_generator: Optional embedding generator (creates default if None)
        """
        self.graph_store = graph_store
        self.vector_index = VectorIndex(embedding_dimension=settings.embedding_dimension)
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.synthesizer = LLMSynthesizer()
        self.formatter = GraphContextFormatter()
        
        # Index chunks for retrieval
        self._chunk_index: Dict[str, Dict[str, Any]] = {}
    
    def index_chunks(self, chunks: List[Any]):
        """Index document chunks for vector search.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        for chunk in chunks:
            # Generate embedding
            vector = self.embedding_generator.generate(chunk.content)
            if vector is not None:
                self.vector_index.add_vector(
                    chunk_id=chunk.id,
                    vector=vector,
                    text=chunk.content,
                    metadata={
                        "document_name": chunk.document_name,
                        "chunk_index": chunk.chunk_index
                    }
                )
                self._chunk_index[chunk.id] = {
                    "content": chunk.content,
                    "document_name": chunk.document_name
                }
        
        logger.info(f"Indexed {len(chunks)} chunks for vector search")
    
    def _extract_entity_ids_from_text(self, text: str) -> List[str]:
        """Extract potential entity IDs from text by matching to graph nodes.
        
        Args:
            text: Text to search
            
        Returns:
            List of matching node IDs
        """
        entity_ids = []
        text_lower = text.lower()
        
        # Search all nodes by name
        for node in self.graph_store.get_stats().node_types.keys():
            # This is a simplified approach - in production would use fuzzy matching
            found_nodes = self.graph_store.find_nodes_by_name(text_lower)
            entity_ids.extend([n.id for n in found_nodes])
        
        return list(set(entity_ids))  # Deduplicate
    
    def retrieve(
        self,
        request: QueryRequest
    ) -> QueryResponse:
        """Perform hybrid retrieval and generate answer.
        
        Args:
            request: Query request with question and parameters
            
        Returns:
            QueryResponse with answer and context
        """
        start_time = time.time()
        question = request.question
        
        logger.info(f"Processing query: {question[:100]}...")
        
        # Step 1: Vector search to find relevant chunks
        query_vector = self.embedding_generator.generate(question)
        if query_vector is None:
            return QueryResponse(
                answer="Unable to process query: embedding generation failed",
                question=question,
                retrieved_context=RetrievedContext(),
                confidence=0.0,
                query_time_ms=0
            )
        
        vector_results = self.vector_index.search(query_vector, top_k=request.top_k)
        
        # Step 2: Identify anchor entities from results
        anchor_node_ids: Set[str] = set()
        anchor_chunks: List[str] = []
        
        for chunk_id, score, text in vector_results:
            anchor_chunks.append(chunk_id)
            
            # Find entities mentioned in this chunk
            mentioned = self._extract_entity_ids_from_text(text)
            anchor_node_ids.update(mentioned)
            
            # Also check for entity name mentions
            nodes, _ = self.graph_store.get_stats()
            for node in self.graph_store.find_nodes_by_name(text):
                anchor_node_ids.add(node.id)
        
        # Step 3: Extract subgraph around anchor nodes
        if not anchor_node_ids:
            # Fallback: use graph stats to find relevant nodes
            all_nodes = []
            for ntype in ["System", "Asset", "Person"]:
                all_nodes.extend(self.graph_store.find_nodes_by_type(ntype))
            anchor_node_ids = {n.id for n in all_nodes[:10]}
        
        # Get neighbors up to max_hops
        context_nodes: Set[str] = set()
        context_edges: List[GraphEdge] = []
        
        for node_id in anchor_node_ids:
            nodes, edges = self.graph_store.get_neighbors_hub(
                node_id, 
                depth=request.max_hops
            )
            context_nodes.update(n.id for n in nodes)
            context_edges.extend(edges)
        
        # Get full subgraph
        subgraph_nodes, subgraph_edges = self.graph_store.get_subgraph(
            list(anchor_node_ids),
            depth=request.max_hops
        )
        
        # Merge results
        all_node_ids = set(n.id for n in subgraph_nodes)
        all_node_ids.update(context_nodes)
        
        final_nodes = []
        for nid in all_node_ids:
            node = self.graph_store.get_node(nid)
            if node:
                final_nodes.append(node)
        
        # Deduplicate edges
        seen_edges = set()
        final_edges = []
        for edge in list(subgraph_edges) + context_edges:
            edge_key = (edge.source_id, edge.target_id, edge.relationship_type)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                final_edges.append(edge)
        
        # Step 4: Format context into text blocks
        text_blocks = self.formatter.format_subgraph(final_nodes, final_edges)
        
        # Step 5: Build context and synthesize answer
        context = RetrievedContext(
            nodes=final_nodes,
            edges=final_edges,
            text_blocks=text_blocks,
            anchor_entities=list(anchor_node_ids)
        )
        
        answer, reasoning_steps, confidence = self.synthesizer.synthesize(
            question=question,
            context=context,
            include_reasoning=request.include_reasoning
        )
        
        # Get source documents
        sources = list(set(
            self._chunk_index.get(cid, {}).get("document_name", "Unknown")
            for cid in anchor_chunks
        ))
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            question=question,
            retrieved_context=context,
            sources=sources,
            reasoning_steps=reasoning_steps if request.include_reasoning else [],
            confidence=confidence,
            query_time_ms=query_time_ms
        )
    
    def get_visualization_data(
        self,
        node_ids: List[str],
        depth: int = 1
    ) -> GraphVisualizationData:
        """Get graph data formatted for visualization.
        
        Args:
            node_ids: Center node IDs
            depth: Traversal depth
            
        Returns:
            GraphVisualizationData for frontend
        """
        nodes, edges = self.graph_store.get_subgraph(node_ids, depth)
        
        viz_nodes = [
            GraphVisualizationNode(
                id=n.id,
                label=n.name,
                type=n.type
            )
            for n in nodes
        ]
        
        viz_edges = [
            GraphVisualizationEdge(
                source=e.source_id,
                target=e.target_id,
                label=e.relationship_type,
                weight=e.weight or 1.0
            )
            for e in edges
        ]
        
        return GraphVisualizationData(
            nodes=viz_nodes,
            edges=viz_edges,
            layout_type="force"
        )
