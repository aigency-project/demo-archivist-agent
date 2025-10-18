# Archivist Agent - Historical Case Specialist for the Detective Agency

The Archivist Agent is a specialized AI assistant built with the Aigency framework that serves as the historical memory and cold case specialist for the Detective Agency. This agent uses advanced RAG (Retrieval-Augmented Generation) technology to search through archived case files and identify patterns, connections, and precedents that can help solve current investigations.

## 🕵️ System Architecture

### Core Agent

**`archivist_agent`** - The Historical Archivist
- Searches the agency's Central Knowledge Base using RAG technology
- Identifies patterns and connections across historical cases
- Provides detailed analysis based on archived evidence and case files
- Specializes in cold case analysis and historical precedent research
- Maintains a methodical, precise approach to investigation

### RAG System Components

- **PDF Document Processing**: Automatically processes PDF case files from the archives
- **Vector Embeddings**: Uses HuggingFace embeddings for semantic search
- **Chroma Vector Database**: Persistent storage for document embeddings
- **Local Generation**: Uses Flan-T5 model for response generation

## 🚀 How to Run

### Prerequisites

1. Docker and Docker Compose installed
2. Environment variables configured in `.env`:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   ```
3. PDF case files placed in `archivist_agent/agent/pdfs/` directory

### Run

```bash
# From the archivist_agent directory
docker-compose up --build
```

### Access Ports

- **Archivist Agent**: http://localhost:8082
- **A2A Inspector**: http://localhost:6007

## 📁 Case Files

The system comes with sample case files in the `pdfs` directory:
- 14 historical case files (007O.pdf through 301F.pdf)
- Each file contains detailed case information, evidence, and investigation notes
- The RAG system automatically processes these files on startup

## 💼 Use Cases

### 1. Pattern Analysis
Find similar cases and identify recurring modus operandi.

**Example Query:**
```
"Find cases with a similar M.O. involving high-tech jewel heists."
```

### 2. Historical Precedent Research
Search for connections between suspects, locations, or methods.

**Example Query:**
```
"What do the archives say about the alias 'The Maestro'?"
```

### 3. Cold Case Investigation
Retrieve specific case files and cross-reference with new evidence.

**Example Query:**
```
"Pull up Case #78B and look for connections to corporate espionage."
```

## 🔧 Agent Configuration

### Archivist Agent Skills

1. **Analyze Patterns** - Identifies modus operandi and conceptual links across cases
2. **Retrieve Case Files** - Searches the knowledge base for specific information
3. **Synthesize Findings** - Creates coherent reports and hypotheses from archival data

### Technical Specifications

- **Model**: Gemini 2.0 Flash (configurable to use Ollama/Llama)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Generation Model**: google/flan-t5-small
- **Chunk Size**: 800 characters with 150 character overlap
- **Retrieval**: Top 3 most relevant documents per query

## 📊 Monitoring and Observability

- **A2A Inspector**: Agent inspection tools at http://localhost:6007
- **Comprehensive Logging**: Detailed logs for RAG operations and document processing
- **Error Recovery**: Automatic vectorstore recovery and rebuilding capabilities

## 🔍 Interaction Examples

### Historical Pattern Analysis
```
User: "I'm investigating a series of art thefts. Are there similar cases in the archives?"

Archivist Agent:
1. Searches archives for art theft patterns
2. Identifies Case #142J - "The Silent Canvas" with similar M.O.
3. Reports: "Case #142J shows identical entry methods and target selection. The perpetrator used sonic devices to disable alarms, matching your current case."
4. Provides specific evidence connections and suspect profiles
```

### Cold Case Cross-Reference
```
User: "We found new DNA evidence in the OmniCorp break-in. Any historical connections?"

Archivist Agent:
1. Queries archives for OmniCorp-related cases
2. Cross-references DNA evidence patterns
3. Reports: "Three archived cases (92A, 199M, 250N) involve OmniCorp facilities. Case #92A contains similar DNA evidence patterns from 2019."
4. Synthesizes timeline and suspect progression
```

## 🛠️ Technical Features

### RAG System Capabilities
- **Automatic PDF Processing**: Loads and processes PDF documents using PyMuPDF and UnstructuredPDF loaders
- **Intelligent Chunking**: Recursive text splitting optimized for case file structure
- **Persistent Storage**: Chroma vectorstore with automatic persistence and recovery
- **Error Handling**: Robust error recovery with automatic database rebuilding
- **Semantic Search**: Advanced similarity search using transformer embeddings

### Development Features
- **Hot Reload**: Automatic restart on code changes during development
- **Modular Architecture**: Clean separation between RAG system and agent logic
- **Configurable Models**: Easy switching between Gemini and local Ollama models
- **Comprehensive Logging**: Detailed operation tracking and debugging support

## 📝 Adding New Case Files

To add new case files to the archive:

1. Place PDF files in `archivist_agent/agent/pdfs/`
2. Restart the agent to rebuild the vectorstore
3. The system will automatically process and index new documents

## 🔐 Security and Privacy

- All processing happens locally within the Docker environment
- No external data transmission for RAG operations
- Case files remain within the secure container environment
- Configurable to use local models (Ollama) for complete air-gapped operation

## 🛠️ Extensibility

The system can be extended with:

- **Additional Document Types**: Support for Word docs, images, and other formats
- **Advanced Analytics**: Statistical analysis of case patterns and trends
- **Integration APIs**: Connect to external case management systems
- **Multi-language Support**: Process case files in multiple languages
- **Real-time Updates**: Live indexing of new case files as they're added