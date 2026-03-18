from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import shutil
import logging
from pathlib import Path

from rag_pipeline import RAGPipeline, RAGResponse
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Q&A System",
    description="A Retrieval-Augmented Generation system for document Q&A",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    search_type: str = "similarity"
    k: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    relevance_scores: List[float]
    query: str
    response_time: float

class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    processing_time: float
    stats: Dict[str, Any]

class PipelineStats(BaseModel):
    llm_model: str
    embedding_model: str
    retrieval_k: int
    temperature: float
    vector_store: Dict[str, Any]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "RAG system is running"}

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "RAG Document Q&A System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "ingest": "/ingest",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG pipeline with a question."""
    try:
        logger.info(f"Received query: {request.question}")
        
        # Process query
        response = rag_pipeline.query(
            question=request.question,
            search_type=request.search_type
        )
        
        # Format sources
        sources = []
        for i, doc in enumerate(response.source_documents):
            sources.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": response.relevance_scores[i] if i < len(response.relevance_scores) else 0.0
            })
        
        return QueryResponse(
            answer=response.answer,
            sources=sources,
            relevance_scores=response.relevance_scores,
            query=response.query,
            response_time=response.response_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ingest documents endpoint
@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None)
):
    """Ingest uploaded documents into the RAG pipeline."""
    try:
        logger.info(f"Received {len(files)} files for ingestion")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files
        file_paths = []
        for file in files:
            # Check file type
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in ['.pdf', '.txt', '.md']:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt, .md"
                )
            
            # Save file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)
        
        # Update settings if provided
        if chunk_size:
            rag_pipeline.update_settings(chunk_size=chunk_size)
        if chunk_overlap:
            rag_pipeline.update_settings(chunk_overlap=chunk_overlap)
        
        # Ingest documents
        result = rag_pipeline.ingest_documents(file_paths=file_paths)
        
        # Schedule cleanup
        background_tasks.add_task(shutil.rmtree, temp_dir)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return IngestResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ingest directory endpoint
@app.post("/ingest-directory", response_model=IngestResponse)
async def ingest_directory(
    background_tasks: BackgroundTasks,
    directory_path: str = Form(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None)
):
    """Ingest all documents from a directory."""
    try:
        logger.info(f"Ingesting documents from directory: {directory_path}")
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")
        
        # Update settings if provided
        if chunk_size:
            rag_pipeline.update_settings(chunk_size=chunk_size)
        if chunk_overlap:
            rag_pipeline.update_settings(chunk_overlap=chunk_overlap)
        
        # Ingest documents
        result = rag_pipeline.ingest_documents(directory_path=directory_path)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return IngestResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting directory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get relevant documents endpoint
@app.post("/retrieve")
async def retrieve_documents(
    question: str = Form(...),
    k: Optional[int] = Form(None)
):
    """Retrieve relevant documents without generating an answer."""
    try:
        documents = rag_pipeline.get_relevant_documents(question, k=k)
        
        # Format documents
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "question": question,
            "documents": formatted_docs,
            "count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get pipeline statistics
@app.get("/stats", response_model=PipelineStats)
async def get_pipeline_stats():
    """Get statistics about the RAG pipeline."""
    try:
        stats = rag_pipeline.get_pipeline_stats()
        return PipelineStats(**stats)
        
    except Exception as e:
        logger.error(f"Error getting pipeline stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete vector store
@app.delete("/vector-store")
async def delete_vector_store():
    """Delete the entire vector store."""
    try:
        rag_pipeline.vector_store.delete_vector_store()
        rag_pipeline.qa_chain = None
        
        return {"status": "success", "message": "Vector store deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Rebuild vector store
@app.post("/rebuild")
async def rebuild_vector_store(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(None),
    directory_path: Optional[str] = Form(None)
):
    """Rebuild the vector store from scratch."""
    try:
        if not files and not directory_path:
            raise HTTPException(
                status_code=400, 
                detail="Either files or directory_path must be provided"
            )
        
        if files:
            # Process uploaded files
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            for file in files:
                file_extension = Path(file.filename).suffix.lower()
                if file_extension not in ['.pdf', '.txt', '.md']:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file type: {file_extension}"
                    )
                
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths.append(file_path)
            
            result = rag_pipeline.ingest_documents(file_paths=file_paths)
            background_tasks.add_task(shutil.rmtree, temp_dir)
        
        else:
            result = rag_pipeline.ingest_documents(directory_path=directory_path)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {"status": "success", "message": "Vector store rebuilt successfully", **result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebuilding vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Update settings
@app.patch("/settings")
async def update_settings(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    retrieval_k: Optional[int] = None,
    temperature: Optional[float] = None
):
    """Update pipeline settings."""
    try:
        updates = {}
        if chunk_size is not None:
            updates["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            updates["chunk_overlap"] = chunk_overlap
        if retrieval_k is not None:
            updates["retrieval_k"] = retrieval_k
        if temperature is not None:
            updates["temperature"] = temperature
        
        rag_pipeline.update_settings(**updates)
        
        return {"status": "success", "updated_settings": updates}
        
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
