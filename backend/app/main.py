"""FastAPI Application Entry Point for GraphRAG Engine.

This module provides the main API endpoints for:
- Document ingestion and processing
- Knowledge graph queries
- System health and statistics
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings, get_settings
from .schemas import (
    QueryRequest, QueryResponse, IngestionResponse,
    HealthStatus, GraphStats, DocumentChunk,
    GraphVisualizationData
)
from .engine.parser import DocumentParser
from .engine.extractor import EntityExtractor
from .engine.graph_store import GraphStore
from .engine.retriever import HybridGraphRetriever

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Global State (Application Scope)
# =============================================================================

# GraphRAG Components
graph_store: Optional[GraphStore] = None
retriever: Optional[HybridGraphRetriever] = None
parser: Optional[DocumentParser] = None
extractor: Optional[EntityExtractor] = None


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global graph_store, retriever, parser, extractor
    
    logger.info("Starting GraphRAG Engine...")
    
    # Initialize components
    graph_store = GraphStore()
    parser = DocumentParser()
    extractor = EntityExtractor()
    retriever = HybridGraphRetriever(graph_store=graph_store)
    
    # Create data directory if it doesn't exist
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Load existing documents if any (skip in test/dev without real API key)
    # For production, this can be enabled by setting DATA_DIR and a valid OPENAI_API_KEY
    if settings.has_openai_key and settings.openai_api_key != "sk-placeholder-key":
        try:
            await load_existing_documents(data_dir)
        except Exception as e:
            logger.warning(f"Could not load existing documents: {e}")
    else:
        logger.info("Skipping auto-load of documents (no valid API key configured)")
    
    logger.info("GraphRAG Engine started successfully")
    logger.info(f"  - Neo4j: {'Enabled' if graph_store.is_neo4j_enabled else 'Disabled'}")
    logger.info(f"  - OpenAI: {'Configured' if settings.has_openai_key else 'Not configured'}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down GraphRAG Engine...")
    graph_store = None
    retriever = None


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="GraphRAG Engine API",
    description="Knowledge Graph-Powered AI Retrieval System",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

async def load_existing_documents(data_dir: Path):
    """Load and process existing markdown files in data directory.
    
    Args:
        data_dir: Directory containing markdown files
    """
    if not data_dir.exists():
        return
    
    md_files = list(data_dir.glob("*.md"))
    if not md_files:
        return
    
    logger.info(f"Loading {len(md_files)} existing documents...")
    
    for md_file in md_files:
        try:
            await process_file(md_file)
        except Exception as e:
            logger.error(f"Failed to load {md_file}: {e}")


async def process_file(file_path: Path) -> IngestionResponse:
    """Process a single markdown file.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        IngestionResponse with processing results
    """
    global graph_store, parser, extractor, retriever
    
    if graph_store is None or parser is None or extractor is None or retriever is None:
        raise RuntimeError("Components not initialized")
    
    try:
        # Parse document
        chunks = parser.parse_file(file_path)
        
        # Index chunks for vector search
        retriever.index_chunks(chunks)
        
        # Extract entities and relationships
        total_nodes = 0
        total_edges = 0
        errors = []
        
        for chunk in chunks:
            result = extractor.extract(
                chunk_content=chunk.content,
                chunk_id=chunk.id,
                document_name=chunk.document_name
            )
            
            nodes_added, edges_added = graph_store.add_extraction(
                result.nodes, result.edges
            )
            total_nodes += nodes_added
            total_edges += edges_added
        
        status = "success" if not errors else "partial"
        message = f"Processed {len(chunks)} chunks, extracted {total_nodes} entities and {total_edges} relationships"
        
        return IngestionResponse(
            document_id=file_path.stem,
            document_name=file_path.name,
            chunks_processed=len(chunks),
            nodes_extracted=total_nodes,
            edges_extracted=total_edges,
            status=status,
            message=message,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return IngestionResponse(
            document_id=file_path.stem,
            document_name=file_path.name,
            chunks_processed=0,
            nodes_extracted=0,
            edges_extracted=0,
            status="failed",
            message=str(e),
            errors=[str(e)]
        )


# =============================================================================
# Health and Statistics Endpoints
# =============================================================================

@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    """Check system health status.
    
    Returns:
        HealthStatus with component states
    """
    global graph_store, retriever
    
    if graph_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = graph_store.get_stats()
    
    return HealthStatus(
        status="healthy" if graph_store else "unhealthy",
        openai_configured=settings.has_openai_key,
        neo4j_configured=settings.is_neo4j_enabled,
        graph_node_count=stats.total_nodes,
        graph_edge_count=stats.total_edges,
        vector_index_size=len(retriever.vector_index) if retriever else 0
    )


@app.get("/stats", response_model=GraphStats, tags=["System"])
async def get_graph_stats():
    """Get knowledge graph statistics.
    
    Returns:
        GraphStats with node/edge counts and distributions
    """
    global graph_store
    
    if graph_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return graph_store.get_stats()


# =============================================================================
# Document Ingestion Endpoints
# =============================================================================

@app.post("/ingest/file", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Markdown file to ingest")
):
    """Ingest a single markdown file.
    
    Args:
        file: Uploaded markdown file
        
    Returns:
        IngestionResponse with processing results
    """
    global parser, extractor, graph_store, retriever
    
    if graph_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Validate file type
    if not file.filename.endswith(".md"):
        raise HTTPException(
            status_code=400, 
            detail="Only markdown files (.md) are supported"
        )
    
    # Create data directory if needed
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = data_dir / file.filename
    try:
        content = await file.read()
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Process the file
    result = await process_file(file_path)
    
    return result


@app.post("/ingest/text", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_text(
    text: str = Query(..., description="Markdown text content"),
    document_name: str = Query("document", description="Document identifier")
):
    """Ingest raw markdown text.
    
    Args:
        text: Markdown text content
        document_name: Document identifier
        
    Returns:
        IngestionResponse with processing results
    """
    global parser, extractor, graph_store, retriever
    
    if graph_store is None or parser is None or extractor is None or retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content is empty")
    
    try:
        # Parse text
        chunks = parser.parse_text(text, document_name)
        
        # Index chunks
        retriever.index_chunks(chunks)
        
        # Extract entities
        total_nodes = 0
        total_edges = 0
        errors = []
        
        for chunk in chunks:
            result = extractor.extract(
                chunk_content=chunk.content,
                chunk_id=chunk.id,
                document_name=document_name
            )
            
            nodes_added, edges_added = graph_store.add_extraction(
                result.nodes, result.edges
            )
            total_nodes += nodes_added
            total_edges += edges_added
        
        return IngestionResponse(
            document_id=document_name,
            document_name=document_name,
            chunks_processed=len(chunks),
            nodes_extracted=total_nodes,
            edges_extracted=total_edges,
            status="success",
            message=f"Processed {len(chunks)} chunks successfully"
        )
        
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Query Endpoints
# =============================================================================

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_knowledge_graph(request: QueryRequest):
    """Query the knowledge graph using hybrid retrieval.
    
    This endpoint performs:
    1. Vector similarity search for anchor entities
    2. Multi-hop graph traversal
    3. LLM synthesis for final answer
    
    Args:
        request: QueryRequest with question and parameters
        
    Returns:
        QueryResponse with answer and context
    """
    global retriever
    
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        response = retriever.retrieve(request)
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/visualization", response_model=GraphVisualizationData, tags=["Query"])
async def get_graph_visualization(
    node_ids: str = Query(..., description="Comma-separated node IDs"),
    depth: int = Query(default=1, ge=1, le=3, description="Traversal depth")
):
    """Get graph data for visualization.
    
    Args:
        node_ids: Comma-separated list of center node IDs
        depth: Traversal depth (1-3)
        
    Returns:
        GraphVisualizationData for frontend rendering
    """
    global retriever
    
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    node_id_list = [nid.strip() for nid in node_ids.split(",") if nid.strip()]
    
    if not node_id_list:
        raise HTTPException(status_code=400, detail="No node IDs provided")
    
    try:
        return retriever.get_visualization_data(node_id_list, depth)
    except Exception as e:
        logger.error(f"Visualization data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Graph Management Endpoints
# =============================================================================

@app.delete("/graph/clear", tags=["Graph"])
async def clear_graph():
    """Clear all nodes and edges from the graph.
    
    Returns:
        Confirmation message
    """
    global graph_store, retriever
    
    if graph_store is None:
        raise HTTPException(status_code=503, detail="Graph store not initialized")
    
    graph_store.clear()
    if retriever:
        retriever.vector_index.clear()
        retriever._chunk_index.clear()
    
    return {"message": "Graph cleared successfully"}


@app.get("/graph/export", tags=["Graph"])
async def export_graph():
    """Export the entire graph as JSON.
    
    Returns:
        Complete graph with nodes and edges
    """
    global graph_store
    
    if graph_store is None:
        raise HTTPException(status_code=503, detail="Graph store not initialized")
    
    return graph_store.export_graph()


@app.get("/graph/nodes/{node_id}", tags=["Graph"])
async def get_node(node_id: str):
    """Get a specific node by ID.
    
    Args:
        node_id: Node identifier
        
    Returns:
        Node details or 404
    """
    global graph_store
    
    if graph_store is None:
        raise HTTPException(status_code=503, detail="Graph store not initialized")
    
    node = graph_store.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    return node


@app.get("/graph/nodes/type/{node_type}", tags=["Graph"])
async def get_nodes_by_type(node_type: str):
    """Get all nodes of a specific type.
    
    Args:
        node_type: Type of nodes (Person, System, Asset, etc.)
        
    Returns:
        List of matching nodes
    """
    global graph_store
    
    if graph_store is None:
        raise HTTPException(status_code=503, detail="Graph store not initialized")
    
    return graph_store.find_nodes_by_type(node_type)


@app.get("/graph/search", tags=["Graph"])
async def search_nodes(
    query: str = Query(..., description="Search query for node names"),
    limit: int = Query(default=10, ge=1, le=100, description="Max results")
):
    """Search nodes by name.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of matching nodes
    """
    global graph_store
    
    if graph_store is None:
        raise HTTPException(status_code=503, detail="Graph store not initialized")
    
    results = graph_store.find_nodes_by_name(query)
    return results[:limit]


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information.
    
    Returns:
        API metadata
    """
    return {
        "name": "GraphRAG Engine API",
        "version": "0.1.0",
        "description": "Knowledge Graph-Powered AI Retrieval System",
        "docs_url": "/docs",
        "health_url": "/health",
        "stats_url": "/stats"
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=True
    )
