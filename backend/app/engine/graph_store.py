"""Graph Store Implementation for GraphRAG Engine.

This module provides a dual-mode graph storage system:
1. In-memory NetworkX graph for fast operations
2. Neo4j connection for production deployments

The system automatically selects the appropriate backend based on configuration.
"""

import json
import logging
import threading
from typing import List, Optional, Dict, Any, Set, Tuple, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager
import hashlib

import networkx as nx

from ..schemas import GraphNode, GraphEdge, GraphStats
from ..config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Neo4j Client (Optional Backend)
# =============================================================================

class Neo4jClient:
    """Neo4j database client for persistent graph storage.
    
    This is an optional backend that provides ACID transactions
    and Cypher query capabilities.
    """
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j client.
        
        Args:
            uri: Neo4j bolt URI
            username: Database username
            password: Database password
        """
        self.uri = uri
        self._driver = None
        self._username = username
        self._password = password
        self._init_driver()
    
    def _init_driver(self):
        """Initialize the Neo4j driver."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self._username, self._password),
                max_connection_lifetime=3600
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ImportError:
            logger.warning("Neo4j driver not installed. Run: pip install neo4j")
            self._driver = None
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = None
    
    @property
    def is_connected(self) -> bool:
        """Check if driver is connected."""
        if self._driver is None:
            return False
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False
    
    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")
        with self._driver.session() as ses:
            yield ses
    
    def add_node(self, node: GraphNode) -> bool:
        """Add a node to Neo4j.
        
        Args:
            node: Node to add
            
        Returns:
            True if successful
        """
        try:
            with self.session() as ses:
                ses.run(
                    """
                    MERGE (n:Node {id: $id})
                    SET n.name = $name,
                        n.type = $type,
                        n.properties = $props
                    """,
                    id=node.id,
                    name=node.name,
                    type=node.type,
                    props=json.dumps(node.properties.model_dump())
                )
            return True
        except Exception as e:
            logger.error(f"Failed to add node to Neo4j: {e}")
            return False
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to Neo4j.
        
        Args:
            edge: Edge to add
            
        Returns:
            True if successful
        """
        try:
            with self.session() as ses:
                ses.run(
                    """
                    MATCH (s:Node {id: $source_id})
                    MATCH (t:Node {id: $target_id})
                    MERGE (s)-[r:RELATES {type: $rel_type}]->(t)
                    SET r.weight = $weight,
                        r.description = $desc,
                        r.properties = $props
                    """,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    rel_type=edge.relationship_type,
                    weight=edge.weight,
                    desc=edge.description,
                    props=json.dumps(edge.properties)
                )
            return True
        except Exception as e:
            logger.error(f"Failed to add edge to Neo4j: {e}")
            return False
    
    def get_neighbors(
        self, 
        node_id: str, 
        depth: int = 1,
        edge_types: Optional[List[str]] = None
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get neighboring nodes up to specified depth.
        
        Args:
            node_id: Starting node ID
            depth: Traversal depth
            edge_types: Filter by edge types
            
        Returns:
            Tuple of (nodes, edges)
        """
        try:
            with self.session() as ses:
                rel_filter = ""
                if edge_types:
                    types_str = " | ".join(edge_types)
                    rel_filter = f"WHERE type(r) IN [{types_str}]"
                
                result = ses.run(
                    f"""
                    MATCH path = (start:Node {{id: $node_id}})
                                -[r*1..{depth}]->
                                (end:Node)
                    {rel_filter}
                    RETURN start, relationships(path) as rels, end
                    """,
                    node_id=node_id
                )
                
                nodes = []
                edges = []
                seen_nodes = set()
                
                for record in result:
                    start = record["start"]
                    end = record["end"]
                    rels = record["rels"]
                    
                    for node_data in [start, end]:
                        if node_data["id"] not in seen_nodes:
                            seen_nodes.add(node_data["id"])
                            props = json.loads(node_data.get("properties", "{}"))
                            nodes.append(GraphNode(
                                id=node_data["id"],
                                name=node_data["name"],
                                type=node_data["type"],
                                properties=props
                            ))
                    
                    for rel in rels:
                        edges.append(GraphEdge(
                            source_id=rel.start_node["id"],
                            target_id=rel.end_node["id"],
                            relationship_type=rel.type,
                            weight=rel.get("weight", 1.0),
                            description=rel.get("description")
                        ))
                
                return nodes, edges
                
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return [], []
    
    def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()


# =============================================================================
# In-Memory Graph Store
# =============================================================================

class InMemoryGraphStore:
    """Thread-safe in-memory graph store using NetworkX.
    
    Provides fast graph operations with automatic node deduplication
    and edge validation.
    """
    
    def __init__(self):
        """Initialize the in-memory graph store."""
        self._graph = nx.MultiDiGraph()
        self._node_metadata: Dict[str, Dict[str, Any]] = {}
        self._edge_metadata: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._node_counter: Dict[str, int] = {}  # For deduplication tracking
        
        # Statistics
        self._stats_lock = threading.Lock()
        self._total_nodes_added = 0
        self._total_edges_added = 0
    
    def _safe_add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph (thread-safe).
        
        Args:
            node: Node to add
            
        Returns:
            True if node was added or already exists
        """
        try:
            with self._lock:
                if not self._graph.has_node(node.id):
                    self._graph.add_node(
                        node.id,
                        name=node.name,
                        type=node.type
                    )
                    self._node_metadata[node.id] = node.properties.model_dump()
                    self._total_nodes_added += 1
                    return True
                else:
                    # Update existing node metadata
                    self._node_metadata[node.id].update(node.properties.model_dump())
                    return True
        except Exception as e:
            logger.error(f"Failed to add node {node.id}: {e}")
            return False
    
    def _safe_add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the graph (thread-safe).
        
        Args:
            edge: Edge to add
            
        Returns:
            True if edge was added or already exists
        """
        try:
            with self._lock:
                if not self._graph.has_edge(edge.source_id, edge.target_id):
                    edge_id = self._graph.add_edge(
                        edge.source_id,
                        edge.target_id,
                        relationship_type=edge.relationship_type,
                        weight=edge.weight,
                        description=edge.description
                    )
                    # Store metadata with edge key
                    edge_key = (edge.source_id, edge.target_id, 0)
                    self._edge_metadata[edge_key] = edge.properties
                    self._total_edges_added += 1
                    return True
                else:
                    # Update existing edge
                    existing = self._graph[edge.source_id][edge.target_id][0]
                    existing.update({
                        "weight": edge.weight,
                        "description": edge.description
                    })
                    return True
        except Exception as e:
            logger.error(f"Failed to add edge {edge.source_id}->{edge.target_id}: {e}")
            return False
    
    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph.
        
        Args:
            node: Node to add
            
        Returns:
            True if successful
        """
        return self._safe_add_node(node)
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the graph.
        
        Args:
            edge: Edge to add
            
        Returns:
            True if successful
        """
        # Ensure both nodes exist
        if not self._graph.has_node(edge.source_id):
            logger.warning(f"Source node {edge.source_id} not found, creating placeholder")
            placeholder = GraphNode(
                id=edge.source_id,
                name=edge.source_id,
                type="Unknown"
            )
            self._safe_add_node(placeholder)
        
        if not self._graph.has_node(edge.target_id):
            logger.warning(f"Target node {edge.target_id} not found, creating placeholder")
            placeholder = GraphNode(
                id=edge.target_id,
                name=edge.target_id,
                type="Unknown"
            )
            self._safe_add_node(placeholder)
        
        return self._safe_add_edge(edge)
    
    def add_extraction(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge]
    ) -> Tuple[int, int]:
        """Add a batch of nodes and edges from extraction.
        
        Args:
            nodes: List of nodes to add
            edges: List of edges to add
            
        Returns:
            Tuple of (nodes_added, edges_added)
        """
        nodes_added = 0
        edges_added = 0
        
        # Add all nodes first
        for node in nodes:
            if self.add_node(node):
                nodes_added += 1
        
        # Then add all edges
        for edge in edges:
            if self.add_edge(edge):
                edges_added += 1
        
        return nodes_added, edges_added
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID.
        
        Args:
            node_id: Node ID to retrieve
            
        Returns:
            GraphNode if found, None otherwise
        """
        with self._lock:
            if self._graph.has_node(node_id):
                data = self._graph.nodes[node_id]
                metadata = self._node_metadata.get(node_id, {})
                return GraphNode(
                    id=node_id,
                    name=data.get("name", node_id),
                    type=data.get("type", "Unknown"),
                    properties=metadata
                )
            return None
    
    def get_neighbors_hub(
        self, 
        node_id: str, 
        depth: int = 1,
        edge_types: Optional[List[str]] = None
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get neighboring nodes up to specified depth (hub-style traversal).
        
        This performs an index-free adjacency traversal from a starting node,
        collecting all reachable nodes and edges within the depth limit.
        
        Args:
            node_id: Starting node ID
            depth: Maximum traversal depth (1-3)
            edge_types: Optional filter for edge types
            
        Returns:
            Tuple of (nodes list, edges list)
        """
        with self._lock:
            if not self._graph.has_node(node_id):
                logger.warning(f"Node {node_id} not found in graph")
                return [], []
            
            depth = max(1, min(depth, 3))  # Clamp depth to 1-3
            
            reached_nodes: Set[str] = {node_id}
            reached_edges: Set[Tuple[str, str]] = set()
            result_nodes: List[GraphNode] = []
            result_edges: List[GraphEdge] = []
            
            # BFS traversal
            current_level = {node_id}
            
            for _ in range(depth):
                next_level = set()
                
                for current_id in current_level:
                    # Get all successors (outgoing edges)
                    successors = self._graph.successors(current_id)
                    for target_id in successors:
                        edge_data = self._graph[current_id][target_id][0]
                        
                        # Filter by edge type if specified
                        if edge_types:
                            rel_type = edge_data.get("relationship_type", "")
                            if rel_type not in edge_types:
                                continue
                        
                        edge_key = (current_id, target_id)
                        if edge_key not in reached_edges:
                            reached_edges.add(edge_key)
                            result_edges.append(GraphEdge(
                                source_id=current_id,
                                target_id=target_id,
                                relationship_type=edge_data.get("relationship_type", "RELATED"),
                                weight=edge_data.get("weight", 1.0),
                                description=edge_data.get("description")
                            ))
                        
                        if target_id not in reached_nodes:
                            reached_nodes.add(target_id)
                            next_level.add(target_id)
                    
                    # Get all predecessors (incoming edges)
                    predecessors = self._graph.predecessors(current_id)
                    for source_id in predecessors:
                        edge_data = self._graph[source_id][current_id][0]
                        
                        if edge_types:
                            rel_type = edge_data.get("relationship_type", "")
                            if rel_type not in edge_types:
                                continue
                        
                        edge_key = (source_id, current_id)
                        if edge_key not in reached_edges:
                            reached_edges.add(edge_key)
                            result_edges.append(GraphEdge(
                                source_id=source_id,
                                target_id=current_id,
                                relationship_type=edge_data.get("relationship_type", "RELATED"),
                                weight=edge_data.get("weight", 1.0),
                                description=edge_data.get("description")
                            ))
                        
                        if source_id not in reached_nodes:
                            reached_nodes.add(source_id)
                            next_level.add(source_id)
                
                current_level = next_level
            
            # Convert node IDs to GraphNode objects
            for nid in reached_nodes:
                data = self._graph.nodes[nid]
                metadata = self._node_metadata.get(nid, {})
                result_nodes.append(GraphNode(
                    id=nid,
                    name=data.get("name", nid),
                    type=data.get("type", "Unknown"),
                    properties=metadata
                ))
            
            return result_nodes, result_edges
    
    def get_subgraph(
        self, 
        node_ids: List[str], 
        depth: int = 1
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get a subgraph containing specified nodes and their neighbors.
        
        Args:
            node_ids: List of node IDs to include
            depth: Additional depth from each node
            
        Returns:
            Tuple of (nodes, edges) in the subgraph
        """
        all_nodes: Set[str] = set()
        all_edges: List[GraphEdge] = []
        
        for node_id in node_ids:
            nodes, edges = self.get_neighbors_hub(node_id, depth)
            all_nodes.update(n.id for n in nodes)
            all_edges.extend(edges)
        
        # Deduplicate edges
        seen_edges = set()
        unique_edges = []
        for edge in all_edges:
            edge_key = (edge.source_id, edge.target_id, edge.relationship_type)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)
        
        # Get full node objects
        result_nodes = []
        for nid in all_nodes:
            node = self.get_node(nid)
            if node:
                result_nodes.append(node)
        
        return result_nodes, unique_edges
    
    def find_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Find all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to find
            
        Returns:
            List of matching nodes
        """
        with self._lock:
            nodes = []
            for nid in self._graph.nodes():
                data = self._graph.nodes[nid]
                if data.get("type", "").lower() == node_type.lower():
                    metadata = self._node_metadata.get(nid, {})
                    nodes.append(GraphNode(
                        id=nid,
                        name=data.get("name", nid),
                        type=data.get("type", "Unknown"),
                        properties=metadata
                    ))
            return nodes
    
    def find_nodes_by_name(self, name_query: str) -> List[GraphNode]:
        """Find nodes by name (case-insensitive partial match).
        
        Args:
            name_query: Name to search for
            
        Returns:
            List of matching nodes
        """
        with self._lock:
            nodes = []
            query = name_query.lower()
            for nid in self._graph.nodes():
                data = self._graph.nodes[nid]
                name = data.get("name", "").lower()
                if query in name:
                    metadata = self._node_metadata.get(nid, {})
                    nodes.append(GraphNode(
                        id=nid,
                        name=data.get("name", nid),
                        type=data.get("type", "Unknown"),
                        properties=metadata
                    ))
            return nodes
    
    def get_stats(self) -> GraphStats:
        """Get graph statistics.
        
        Returns:
            GraphStats object with current statistics
        """
        with self._lock:
            node_types: Dict[str, int] = {}
            rel_types: Dict[str, int] = {}
            
            for nid in self._graph.nodes():
                ntype = self._graph.nodes[nid].get("type", "Unknown")
                node_types[ntype] = node_types.get(ntype, 0) + 1
            
            for source, target, data in self._graph.edges(data=True):
                rtype = data.get("relationship_type", "RELATED")
                rel_types[rtype] = rel_types.get(rtype, 0) + 1
            
            # Count documents (rough estimate from metadata)
            documents = set()
            for metadata in self._node_metadata.values():
                if isinstance(metadata, dict) and "source_chunk_id" in metadata:
                    chunk_id = metadata["source_chunk_id"]
                    if chunk_id and "_" in str(chunk_id):
                        doc_name = str(chunk_id).split("_")[1]
                        documents.add(doc_name)
            
            return GraphStats(
                total_nodes=self._graph.number_of_nodes(),
                total_edges=self._graph.number_of_edges(),
                node_types=node_types,
                relationship_types=rel_types,
                documents_processed=len(documents),
                avg_nodes_per_document=(
                    self._graph.number_of_nodes() / len(documents) 
                    if documents else 0
                )
            )
    
    def clear(self):
        """Clear all nodes and edges from the graph."""
        with self._lock:
            self._graph.clear()
            self._node_metadata.clear()
            self._edge_metadata.clear()
            self._total_nodes_added = 0
            self._total_edges_added = 0
            logger.info("Graph cleared")
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the entire graph as JSON-serializable dict.
        
        Returns:
            Dictionary with nodes and edges
        """
        with self._lock:
            nodes = []
            for nid in self._graph.nodes():
                data = self._graph.nodes[nid]
                nodes.append({
                    "id": nid,
                    "name": data.get("name", nid),
                    "type": data.get("type", "Unknown"),
                    "properties": self._node_metadata.get(nid, {})
                })
            
            edges = []
            for source, target, data in self._graph.edges(data=True):
                edges.append({
                    "source_id": source,
                    "target_id": target,
                    "relationship_type": data.get("relationship_type", "RELATED"),
                    "weight": data.get("weight", 1.0),
                    "description": data.get("description")
                })
            
            return {"nodes": nodes, "edges": edges}


# =============================================================================
# Graph Store Factory
# =============================================================================

class GraphStore:
    """Unified graph store interface with automatic backend selection.
    
    Automatically uses Neo4j if configured, otherwise falls back to
    in-memory NetworkX store.
    """
    
    def __init__(self):
        """Initialize the graph store with appropriate backend."""
        self._memory_store: Optional[InMemoryGraphStore] = None
        self._neo4j_client: Optional[Neo4jClient] = None
        self._use_neo4j = False
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the configured backend."""
        if settings.use_neo4j:
            try:
                self._neo4j_client = Neo4jClient(
                    uri=settings.neo4j_uri,
                    username=settings.neo4j_username,
                    password=settings.neo4j_password
                )
                if self._neo4j_client.is_connected:
                    self._use_neo4j = True
                    logger.info("Using Neo4j backend")
                else:
                    logger.warning("Neo4j connection failed, using in-memory store")
                    self._use_neo4j = False
            except Exception as e:
                logger.warning(f"Neo4j initialization failed: {e}")
                self._use_neo4j = False
        
        # Always create in-memory store as fallback/cache
        self._memory_store = InMemoryGraphStore()
        logger.info("In-memory graph store initialized")
    
    @property
    def is_neo4j_enabled(self) -> bool:
        """Check if Neo4j backend is active."""
        return self._use_neo4j and self._neo4j_client is not None
    
    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph store."""
        # Always add to memory store
        memory_result = self._memory_store.add_node(node)
        
        # Also add to Neo4j if available
        if self._use_neo4j and self._neo4j_client:
            self._neo4j_client.add_node(node)
        
        return memory_result
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the graph store."""
        memory_result = self._memory_store.add_edge(edge)
        
        if self._use_neo4j and self._neo4j_client:
            self._neo4j_client.add_edge(edge)
        
        return memory_result
    
    def add_extraction(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge]
    ) -> Tuple[int, int]:
        """Add extraction results to the store."""
        return self._memory_store.add_extraction(nodes, edges)
    
    def get_neighbors_hub(
        self, 
        node_id: str, 
        depth: int = 1,
        edge_types: Optional[List[str]] = None
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get neighboring nodes up to specified depth."""
        return self._memory_store.get_neighbors_hub(node_id, depth, edge_types)
    
    def get_subgraph(
        self, 
        node_ids: List[str], 
        depth: int = 1
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get subgraph around specified nodes."""
        return self._memory_store.get_subgraph(node_ids, depth)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._memory_store.get_node(node_id)
    
    def find_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Find nodes by type."""
        return self._memory_store.find_nodes_by_type(node_type)
    
    def find_nodes_by_name(self, name_query: str) -> List[GraphNode]:
        """Find nodes by name."""
        return self._memory_store.find_nodes_by_name(name_query)
    
    def get_stats(self) -> GraphStats:
        """Get graph statistics."""
        return self._memory_store.get_stats()
    
    def clear(self):
        """Clear the graph store."""
        self._memory_store.clear()
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the graph."""
        return self._memory_store.export_graph()
