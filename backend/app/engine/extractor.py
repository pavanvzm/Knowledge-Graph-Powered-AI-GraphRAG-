"""Entity and Relationship Extraction Module for GraphRAG.

This module uses OpenAI's function calling to extract structured entities
and relationships from document chunks into typed graph components.
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from ..schemas import GraphNode, GraphEdge, ExtractionResult, NodeProperties
from ..config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Extraction Pydantic Models
# =============================================================================

class ExtractedNode(BaseModel):
    """Schema for extracted node from LLM."""
    id: str = Field(description="Unique identifier for the entity")
    name: str = Field(description="Human-readable name of the entity")
    entity_type: str = Field(description="Type of entity (Person, System, Asset, etc.)")
    description: Optional[str] = Field(None, description="Brief description of the entity")
    properties: Dict[str, Any] = Field(default_factory=dict)


class ExtractedEdge(BaseModel):
    """Schema for extracted relationship from LLM."""
    source_id: str = Field(description="Identifier of the source entity")
    target_id: str = Field(description="Identifier of the target entity")
    relationship_type: str = Field(description="Type of relationship (DEPENDS_ON, OWNS, etc.)")
    description: Optional[str] = Field(None, description="Description of the relationship")
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractionOutput(BaseModel):
    """Schema for complete extraction output."""
    nodes: List[ExtractedNode] = Field(default_factory=list)
    edges: List[ExtractedEdge] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Explanation of extraction decisions")


# =============================================================================
# Extraction Prompt Templates
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """You are an expert knowledge engineer specializing in extracting structured entities and relationships from technical documentation.

Analyze the following document chunk and extract all meaningful entities and their relationships.

ENTITY TYPES TO IDENTIFY:
- Person: Individuals, team members, roles
- System: Software systems, applications, services
- Asset: Infrastructure, databases, servers, tools
- Process: Workflows, procedures, methodologies
- Concept: Ideas, standards, policies, requirements
- Document: Specifications, manuals, tickets

RELATIONSHIP TYPES:
- DEPENDS_ON: System/service requires another to function
- OWNS: Person/team owns or is responsible for a system
- USES: System uses a tool, library, or asset
- CONNECTS_TO: Systems/services are connected or integrated
- SUPPORTS: Person/team supports a system
- IMPLEMENTS: System implements a concept/standard
- CONTAINS: System contains subsystems or components
- AUTHORED_BY: Document was created by a person
- REFERENCES: Document references another document

Return your extraction as a structured JSON object with 'nodes' and 'edges' arrays.

DOCUMENT CHUNK:
{chunk_content}

Extract entities and relationships that are explicitly stated or clearly implied. Use the exact names from the text where possible.

Output JSON format:
{{
  "nodes": [
    {{
      "id": "unique_id_for_entity",
      "name": "Entity Name",
      "entity_type": "System",
      "description": "Brief description",
      "properties": {{}}
    }}
  ],
  "edges": [
    {{
      "source_id": "id_of_source_entity",
      "target_id": "id_of_target_entity",
      "relationship_type": "DEPENDS_ON",
      "description": "Nature of relationship",
      "weight": 1.0
    }}
  ],
  "reasoning": "Brief explanation of extraction choices"
}}"""


ENTITY_EXTRACTION_FUNCTIONS = [
    {
        "name": "extract_entities",
        "description": "Extract structured entities and relationships from the document text",
        "parameters": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique identifier for the entity"},
                            "name": {"type": "string", "description": "Human-readable name"},
                            "entity_type": {"type": "string", "enum": ["Person", "System", "Asset", "Process", "Concept", "Document"]},
                            "description": {"type": "string"},
                            "properties": {"type": "object"}
                        },
                        "required": ["id", "name", "entity_type"]
                    },
                    "description": "List of extracted entities"
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "target_id": {"type": "string"},
                            "relationship_type": {"type": "string"},
                            "description": {"type": "string"},
                            "weight": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["source_id", "target_id", "relationship_type"]
                    },
                    "description": "List of extracted relationships"
                },
                "reasoning": {"type": "string", "description": "Explanation of extraction"}
            },
            "required": ["nodes", "edges"]
        }
    }
]


# =============================================================================
# Entity Extractor Class
# =============================================================================

class EntityExtractor:
    """Extracts structured entities and relationships from text using LLMs.
    
    Supports both raw JSON extraction and function calling modes for
    reliable structured output.
    """
    
    VALID_ENTITY_TYPES = {"Person", "System", "Asset", "Process", "Concept", "Document"}
    VALID_RELATIONSHIP_TYPES = {
        "DEPENDS_ON", "OWNS", "USES", "CONNECTS_TO", "SUPPORTS",
        "IMPLEMENTS", "CONTAINS", "AUTHORED_BY", "REFERENCES"
    }
    
    def __init__(self, openai_client=None):
        """Initialize the entity extractor.
        
        Args:
            openai_client: OpenAI client instance. If None, uses default configuration.
        """
        self._client = openai_client
        self._use_functions = True
        self._extraction_count = 0
    
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
    
    def _validate_extraction(self, extraction: ExtractionOutput) -> ExtractionOutput:
        """Validate and clean extracted entities and relationships.
        
        Args:
            extraction: Raw extraction output
            
        Returns:
            Validated extraction with normalized types
        """
        # Normalize entity types
        for node in extraction.nodes:
            if node.entity_type not in self.VALID_ENTITY_TYPES:
                # Try to map common types
                type_mapping = {
                    "Application": "System",
                    "Service": "System",
                    "Database": "Asset",
                    "Server": "Asset",
                    "Team": "Person",
                    "User": "Person",
                    "Standard": "Concept",
                    "Policy": "Concept",
                    "Workflow": "Process",
                    "Ticket": "Document",
                }
                node.entity_type = type_mapping.get(node.entity_type, "Concept")
        
        # Normalize relationship types (keep uppercase)
        for edge in extraction.edges:
            edge.relationship_type = edge.relationship_type.upper().strip()
            if edge.relationship_type not in self.VALID_RELATIONSHIP_TYPES:
                # Use CONNECTS_TO as fallback for unknown relationships
                logger.debug(f"Unknown relationship type: {edge.relationship_type}, using CONNECTS_TO")
                edge.relationship_type = "CONNECTS_TO"
        
        # Ensure edge weights are valid
        for edge in extraction.edges:
            if edge.weight is None or not (0 <= edge.weight <= 1):
                edge.weight = 1.0
        
        return extraction
    
    def _generate_node_id(self, name: str, entity_type: str, chunk_id: str) -> str:
        """Generate a unique, deterministic node ID.
        
        Args:
            name: Entity name
            entity_type: Entity type
            chunk_id: Source chunk ID
            
        Returns:
            Unique node identifier
        """
        import hashlib
        content = f"{name}:{entity_type}:{chunk_id}".lower()
        hash_digest = hashlib.md5(content.encode()).hexdigest()[:12]
        # Create a clean ID: lowercase, no spaces
        clean_name = re.sub(r'[^a-z0-9]', '_', name.lower())
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')[:20]
        return f"{clean_name}_{hash_digest}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic.
        
        Args:
            prompt: Prompt to send to LLM
            
        Returns:
            LLM response text
        """
        if self.client is None:
            raise RuntimeError("OpenAI client not available")
        
        response = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a precise knowledge extraction assistant. Always output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _call_llm_with_functions(self, chunk_content: str) -> ExtractionOutput:
        """Call LLM using function calling for structured extraction.
        
        Args:
            chunk_content: Document chunk to extract from
            
        Returns:
            Structured extraction output
        """
        if self.client is None:
            raise RuntimeError("OpenAI client not available")
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(chunk_content=chunk_content)
        
        response = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a precise knowledge extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            tools=ENTITY_EXTRACTION_FUNCTIONS,
            tool_choice={"type": "function", "function": {"name": "extract_entities"}},
            temperature=0.1,
            max_tokens=2500
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            return ExtractionOutput(**arguments)
        
        # Fallback: parse raw response
        if message.content:
            return self._parse_json_response(message.content)
        
        raise ValueError("No extraction result from LLM")
    
    def _parse_json_response(self, response_text: str) -> ExtractionOutput:
        """Parse JSON response from LLM.
        
        Args:
            response_text: Raw response text containing JSON
            
        Returns:
            Parsed extraction output
        """
        # Try to extract JSON from markdown code blocks or raw text
        json_match = re.search(
            r'```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*\})',
            response_text
        )
        
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            try:
                data = json.loads(json_str)
                return ExtractionOutput(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to parse JSON: {e}")
        
        # Try to parse entire text as JSON
        try:
            data = json.loads(response_text)
            return ExtractionOutput(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse extraction response: {e}")
            raise ValueError(f"Invalid extraction response: {response_text[:200]}")
    
    def extract(self, chunk_content: str, chunk_id: str, document_name: str) -> ExtractionResult:
        """Extract entities and relationships from a document chunk.
        
        Args:
            chunk_content: Text content of the chunk
            chunk_id: Unique identifier for the chunk
            document_name: Name of the source document
            
        Returns:
            ExtractionResult with extracted nodes and edges
        """
        self._extraction_count += 1
        
        try:
            # Try function calling first, fall back to JSON parsing
            if self._use_functions and self.client:
                try:
                    extraction = self._call_llm_with_functions(chunk_content)
                except Exception as e:
                    logger.warning(f"Function calling failed, falling back to JSON: {e}")
                    self._use_functions = False
                    extraction = self._call_llm(chunk_content)
                    extraction = self._parse_json_response(extraction)
            else:
                prompt = ENTITY_EXTRACTION_PROMPT.format(chunk_content=chunk_content)
                response_text = self._call_llm(prompt)
                extraction = self._parse_json_response(response_text)
            
            # Validate extraction
            extraction = self._validate_extraction(extraction)
            
            # Generate deterministic IDs and create graph objects
            nodes = []
            node_id_map: Dict[str, str] = {}
            
            for extracted_node in extraction.nodes:
                node_id = self._generate_node_id(
                    extracted_node.name,
                    extracted_node.entity_type,
                    chunk_id
                )
                node_id_map[extracted_node.name.lower()] = node_id
                
                nodes.append(GraphNode(
                    id=node_id,
                    name=extracted_node.name,
                    type=extracted_node.entity_type,
                    properties=NodeProperties(
                        description=extracted_node.description,
                        source_chunk_id=chunk_id,
                        metadata=extracted_node.properties
                    )
                ))
            
            # Map relationships to node IDs
            edges = []
            for extracted_edge in extraction.edges:
                # Try to resolve source and target by name or use as-is
                source_id = extracted_edge.source_id
                target_id = extracted_edge.target_id
                
                # Check if source/target are names that need mapping
                if source_id.lower() in node_id_map:
                    source_id = node_id_map[source_id.lower()]
                if target_id.lower() in node_id_map:
                    target_id = node_id_map[target_id.lower()]
                
                edges.append(GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=extracted_edge.relationship_type,
                    weight=extracted_edge.weight or 1.0,
                    description=extracted_edge.description,
                    properties={"source_chunk": chunk_id}
                ))
            
            logger.debug(
                f"Extracted {len(nodes)} nodes and {len(edges)} edges "
                f"from chunk {chunk_id}"
            )
            
            return ExtractionResult(
                nodes=nodes,
                edges=edges,
                source_chunk_id=chunk_id,
                source_document=document_name
            )
            
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk_id}: {e}")
            return ExtractionResult(
                nodes=[],
                edges=[],
                source_chunk_id=chunk_id,
                source_document=document_name
            )
    
    def extract_batch(
        self, 
        chunks: List[Dict[str, str]], 
        batch_size: int = 5
    ) -> List[ExtractionResult]:
        """Extract entities from multiple chunks in batch.
        
        Args:
            chunks: List of dicts with 'content', 'id', 'document_name'
            batch_size: Number of chunks per batch
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                result = self.extract(
                    chunk_content=chunk["content"],
                    chunk_id=chunk["id"],
                    document_name=chunk["document_name"]
                )
                results.append(result)
            
            logger.info(f"Processed batch {i // batch_size + 1}, total: {len(results)}")
        
        return results
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about extraction operations.
        
        Returns:
            Dictionary of extraction statistics
        """
        return {
            "total_extractions": self._extraction_count,
            "function_calling_enabled": self._use_functions
        }
