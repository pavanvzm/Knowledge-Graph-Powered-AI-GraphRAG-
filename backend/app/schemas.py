"""Pydantic models for API request/response schemas."""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Entity and Relationship Models (from extractor)
# =============================================================================

class NodeProperties(BaseModel):
    """Additional properties for a graph node."""
    description: Optional[str] = None
    source_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """Represents a node/vertex in the knowledge graph."""
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(..., description="Unique identifier for the node")
    name: str = Field(..., description="Human-readable name of the entity")
    type: str = Field(..., description="Entity type (e.g., Person, System, Asset)")
    properties: NodeProperties = Field(default_factory=NodeProperties)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties.model_dump()
        }


class GraphEdge(BaseModel):
    """Represents an edge/relationship in the knowledge graph."""
    model_config = ConfigDict(extra="allow")
    
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Type of relationship")
    weight: Optional[float] = Field(default=1.0, description="Edge weight/confidence")
    description: Optional[str] = Field(None, description="Relationship description")
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "description": self.description,
            "properties": self.properties
        }


class ExtractionResult(BaseModel):
    """Result of entity extraction from a document chunk."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    source_chunk_id: str = Field(..., description="ID of the source chunk")
    source_document: str = Field(..., description="Name of the source document")


# =============================================================================
# Document Processing Models
# =============================================================================

class DocumentChunk(BaseModel):
    """A chunk of a document for processing."""
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Text content of the chunk")
    document_name: str = Field(..., description="Source document name")
    chunk_index: int = Field(..., description="Index of chunk in document")
    total_chunks: int = Field(..., description="Total chunks in document")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Metadata for an uploaded document."""
    filename: str
    file_size: int
    chunk_count: int
    uploaded_at: datetime = Field(default_factory=datetime.now)


class IngestionResponse(BaseModel):
    """Response from document ingestion."""
    document_id: str
    document_name: str
    chunks_processed: int
    nodes_extracted: int
    edges_extracted: int
    status: Literal["success", "partial", "failed"]
    message: str
    errors: List[str] = Field(default_factory=list)


# =============================================================================
# Query and Retrieval Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request for a GraphRAG query."""
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    max_hops: int = Field(default=2, ge=1, le=3, description="Max traversal depth")
    include_reasoning: bool = Field(default=True, description="Include reasoning steps")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Which systems does the authentication service depend on?",
                "top_k": 5,
                "max_hops": 2
            }
        }
    )


class RetrievedContext(BaseModel):
    """Retrieved context from graph traversal."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    text_blocks: List[str] = Field(default_factory=list)
    anchor_entities: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response from a GraphRAG query."""
    answer: str = Field(..., description="Generated answer")
    question: str = Field(..., description="Original question")
    retrieved_context: RetrievedContext
    sources: List[str] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    query_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Graph Visualization Models
# =============================================================================

class GraphVisualizationNode(BaseModel):
    """Node data for visualization."""
    id: str
    label: str
    type: str
    x: Optional[float] = None
    y: Optional[float] = None


class GraphVisualizationEdge(BaseModel):
    """Edge data for visualization."""
    source: str
    target: str
    label: str
    weight: float = 1.0


class GraphVisualizationData(BaseModel):
    """Full graph data for frontend visualization."""
    nodes: List[GraphVisualizationNode]
    edges: List[GraphVisualizationEdge]
    layout_type: str = "force"


# =============================================================================
# Status and Health Models
# =============================================================================

class HealthStatus(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    openai_configured: bool
    neo4j_configured: bool
    graph_node_count: int
    graph_edge_count: int
    vector_index_size: int


class GraphStats(BaseModel):
    """Statistics about the knowledge graph."""
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]
    documents_processed: int
    avg_nodes_per_document: float
