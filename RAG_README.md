# LLM-Based Document Q&A System (RAG Implementation)

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, and FastAPI for intelligent document Q&A.

## 🚀 Features

- **📄 Document Processing**: Support for PDF, TXT, and MD files
- **🔍 Intelligent Retrieval**: FAISS-based vector similarity search
- **🤖 Context-Aware Responses**: Integration with OpenAI and Hugging Face LLMs
- **⚡ FastAPI REST API**: Production-ready REST endpoints
- **🐳 Docker Support**: Containerized deployment
- **📊 Analytics**: Pipeline statistics and performance metrics
- **🔄 Multiple Search Strategies**: Similarity, MMR, and score-based retrieval

## 📋 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Document       │    │   Vector        │
│   Upload        │───▶│   Processor      │───▶│   Store (FAISS) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│   User Query    │───▶│   RAG Pipeline   │◀───────────┘
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   LLM Response  │
                       └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-document-qa
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually**
   ```bash
   docker build -t rag-qa .
   docker run -p 8000:8000 --env-file .env rag-qa
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Alternative: Hugging Face API Configuration
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2

# FAISS Configuration
FAISS_INDEX_PATH=./faiss_index
EMBEDDING_MODEL=text-embedding-ada-002

# FastAPI Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4000

# RAG Configuration
RETRIEVAL_K=4
TEMPERATURE=0.7
```

## 📚 API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Core Endpoints

#### 1. Health Check
```http
GET /health
```

#### 2. Ingest Documents
```http
POST /ingest
Content-Type: multipart/form-data

# Upload multiple files
files: [File]
chunk_size: 1000 (optional)
chunk_overlap: 200 (optional)
```

#### 3. Query Documents
```http
POST /query
Content-Type: application/json

{
  "question": "What is artificial intelligence?",
  "search_type": "similarity",
  "k": 4
}
```

#### 4. Get Pipeline Statistics
```http
GET /stats
```

#### 5. Retrieve Documents (No LLM)
```http
POST /retrieve
Content-Type: multipart/form-data

question: "Your question here"
k: 4 (optional)
```

## 🎯 Usage Examples

### Python Client Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Upload documents
files = [("files", open("document.pdf", "rb"))]
response = requests.post(f"{BASE_URL}/ingest", files=files)
print("Ingestion:", response.json())

# Query the system
query_data = {
    "question": "What are the main benefits of RAG?",
    "search_type": "similarity"
}
response = requests.post(
    f"{BASE_URL}/query",
    json=query_data
)
result = response.json()
print("Answer:", result["answer"])
print("Sources:", len(result["sources"]))
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Ingest documents
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "search_type": "similarity"
  }'
```

## 🔍 Search Strategies

### 1. Similarity Search
Default search method using cosine similarity.

### 2. Max Marginal Relevance (MMR)
Provides diverse results by balancing relevance and diversity.

### 3. Similarity with Scores
Returns relevance scores along with documents.

## 📊 Performance Optimization

### Chunking Strategies

- **Recursive Character Splitting**: Maintains context boundaries
- **Token-based Splitting**: Precise token count control
- **Configurable Overlap**: Prevents context loss

### Vector Store Optimization

- **FAISS Indexing**: Efficient similarity search
- **Persistent Storage**: Index persistence across restarts
- **Batch Processing**: Optimized for large document sets

## 🐳 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   export OPENAI_API_KEY="your_key"
   export CHUNK_SIZE=1000
   export RETRIEVAL_K=4
   ```

2. **Docker Deployment**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

3. **Load Balancing**
   - Use nginx or similar for load balancing
   - Configure health checks
   - Set up monitoring

### Scaling Considerations

- **Horizontal Scaling**: Multiple container instances
- **Vector Store**: Consider FAISS GPU version for large datasets
- **Caching**: Redis for query result caching
- **Monitoring**: Prometheus/Grafana for metrics

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Load Testing
```bash
# Using locust
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📈 Monitoring

### Health Checks
- `/health` endpoint for service status
- Docker health checks
- Custom metrics in `/stats`

### Logging
- Structured logging with levels
- Request/response logging
- Error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Issues**: Check environment variables
3. **Memory Issues**: Reduce chunk size for large documents
4. **FAISS Errors**: Check index permissions and disk space

### Debug Mode

Enable debug logging:
```bash
export DEBUG=true
python main.py
```

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API docs at `/docs`

---

Built with ❤️ using LangChain, FAISS, and FastAPI
