# GraphRAG Engine - Knowledge Graph-Powered AI Retrieval

A production-grade GraphRAG (Knowledge Graph-Powered AI Retrieval) system that ingests unstructured markdown documentation, extracts entities and relationships using LLMs, stores them in a graph database, and executes hybrid vector + graph queries to answer complex relational questions.

## 🌟 Features

- **Document Ingestion**: Parse and chunk markdown documents with intelligent section detection
- **Entity Extraction**: OpenAI function calling for structured entity/relationship extraction
- **Graph Storage**: In-memory NetworkX store with optional Neo4j backend
- **Hybrid Retrieval**: Vector similarity + multi-hop graph traversal
- **LLM Synthesis**: Generate comprehensive answers from graph context
- **Interactive UI**: Real-time chat interface with live graph visualization

## 📁 Project Structure

```
graph-rag-engine/
├── backend/
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example          # Configuration template
│   └── app/
│       ├── config.py         # Settings management
│       ├── schemas.py        # Pydantic models
│       ├── main.py           # FastAPI endpoints
│       └── engine/
│           ├── parser.py     # Document chunking
│           ├── extractor.py   # Entity extraction
│           ├── graph_store.py # Graph operations
│           └── retriever.py  # Hybrid retrieval
├── frontend/                 # Next.js 14+ App Router
│   └── src/app/
│       ├── page.tsx         # Main UI
│       └── components/
│           └── GraphVisualization.tsx
└── tests/
    └── test_pipeline.py     # PyTest suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- OpenAI API key (for LLM features)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Running Tests

```bash
cd backend
pip install pytest pytest-asyncio httpx
python -m pytest ../tests/test_pipeline.py -v
```

## 🔧 Configuration

Create a `.env` file in the `backend/` directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Neo4j Configuration (Optional)
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=your-password

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# GraphRAG Configuration
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_HOP_DEPTH=2
```

## 📡 API Endpoints

### Health & Stats
- `GET /` - API information
- `GET /health` - System health check
- `GET /stats` - Graph statistics

### Document Ingestion
- `POST /ingest/file` - Upload and process markdown file
- `POST /ingest/text` - Ingest raw markdown text

### Querying
- `POST /query` - Query the knowledge graph
- `GET /query/visualization` - Get graph data for visualization

### Graph Management
- `DELETE /graph/clear` - Clear all nodes and edges
- `GET /graph/export` - Export entire graph as JSON
- `GET /graph/nodes/{node_id}` - Get specific node
- `GET /graph/nodes/type/{node_type}` - Get nodes by type
- `GET /graph/search?q={query}` - Search nodes by name

## 🔄 Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which systems does the authentication service depend on?",
    "top_k": 5,
    "max_hops": 2
  }'
```

## 🏗️ Architecture

### Document Processing Pipeline

1. **Parser**: Chunks markdown into token-aware segments
2. **Extractor**: Uses OpenAI function calling to extract entities/relationships
3. **Graph Store**: Stores nodes (entities) and edges (relationships) in graph
4. **Retriever**: Performs hybrid search combining vector similarity + graph traversal

### Entity Types

- **Person**: Individuals, teams, roles
- **System**: Applications, services
- **Asset**: Infrastructure, databases
- **Process**: Workflows, procedures
- **Concept**: Standards, policies
- **Document**: Specs, tickets

### Relationship Types

- `DEPENDS_ON` - Service requires another to function
- `OWNS` - Person/team is responsible for
- `USES` - Service uses a tool or asset
- `CONNECTS_TO` - Systems are integrated
- `SUPPORTS` - Team supports a system
- `IMPLEMENTS` - System implements a concept
- `CONTAINS` - System contains components
- `AUTHORED_BY` - Document created by
- `REFERENCES` - Document references another

## 🧪 Test Results

```
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0

tests/test_pipeline.py::TestDocumentParser::test_parse_file PASSED       [  4%]
tests/test_pipeline.py::TestDocumentParser::test_parse_text PASSED       [  9%]
tests/test_pipeline.py::TestDocumentParser::test_empty_text_raises_error PASSED [ 13%]
tests/test_pipeline.py::TestDocumentParser::test_chunk_config_validation PASSED [ 18%]
tests/test_pipeline.py::TestDocumentParser::test_chunk_statistics PASSED [ 22%]
tests/test_pipeline.py::TestGraphStore::test_add_node PASSED             [ 27%]
tests/test_pipeline.py::TestGraphStore::test_add_edge PASSED             [ 31%]
tests/test_pipeline.py::TestGraphStore::test_get_neighbors_hub PASSED    [ 36%]
tests/test_pipeline.py::TestGraphStore::test_get_subgraph PASSED         [ 40%]
tests/test_pipeline.py::TestGraphStore::test_find_nodes_by_type PASSED   [ 45%]
tests/test_pipeline.py::TestGraphStore::test_find_nodes_by_name PASSED   [ 50%]
tests/test_pipeline.py::TestGraphStore::test_graph_stats PASSED          [ 54%]
tests/test_pipeline.py::TestGraphStore::test_clear_graph PASSED          [ 59%]
tests/test_pipeline.py::TestEntityExtractor::test_extraction_with_mock PASSED [ 63%]
tests/test_pipeline.py::TestEntityExtractor::test_id_generation PASSED   [ 68%]
tests/test_pipeline.py::TestEntityExtractor::test_validation_normalizes_types PASSED [ 72%]
tests/test_pipeline.py::TestHybridGraphRetriever::test_visualization_data PASSED [ 77%]
tests/test_pipeline.py::TestAPIEndpoints::test_root_endpoint PASSED      [ 81%]
tests/test_pipeline.py::TestAPIEndpoints::test_health_endpoint PASSED    [ 86%]
tests/test_pipeline.py::TestPipelineIntegration::test_full_ingestion_pipeline PASSED [ 90%]
tests/test_pipeline.py::TestPipelineIntegration::test_graph_traversal_depth PASSED [ 95%]
tests/test_pipeline.py::TestPerformance::test_large_graph_operations PASSED [100%]

======================== 22 passed, 1 warning in 1.68s =========================
```

## 📊 Sample Data

The `backend/data/sample.md` file contains example infrastructure documentation for testing.

## 🔒 Security

- API keys are loaded from environment variables
- Graph operations are wrapped in validation
- No user input is executed directly

## 📝 License

MIT License - See LICENSE file for details.
