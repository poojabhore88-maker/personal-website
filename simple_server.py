#!/usr/bin/env python3
"""
Simple FastAPI Server for RAG Demo - Shows output on localhost
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import time
import json
from typing import List, Dict, Any

# Import our working RAG demo
from working_rag_demo import WorkingRAGPipeline

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
rag_pipeline = WorkingRAGPipeline()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    search_type: str = "similarity"
    k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    question: str
    timing: Dict[str, float]

# Ingest sample documents on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline with sample documents"""
    print("🚀 Starting RAG server...")
    
    # Sample documents
    documents = [
        """
        Artificial Intelligence Fundamentals
        
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines capable of performing tasks that typically require human intelligence. 
        These tasks include learning, reasoning, problem-solving, perception, and language understanding.
        
        The field of AI was founded in 1956 at a conference at Dartmouth College. Since then, 
        it has gone through multiple periods of optimism and pessimism, known as "AI winters" 
        and "AI summers."
        
        Modern AI systems use various approaches including machine learning, deep learning, 
        natural language processing, computer vision, and robotics.
        """,
        
        """
        Machine Learning Applications
        
        Machine Learning (ML) is a subset of AI that enables systems to learn and improve 
        from experience without being explicitly programmed. ML algorithms use training data 
        to make predictions or decisions.
        
        Common ML applications include:
        - Recommendation systems (Netflix, Amazon)
        - Spam email filtering
        - Credit card fraud detection
        - Medical diagnosis
        - Stock market prediction
        - Autonomous vehicles
        
        The three main types of machine learning are:
        1. Supervised Learning (learning with labeled data)
        2. Unsupervised Learning (finding patterns in unlabeled data)
        3. Reinforcement Learning (learning through trial and error)
        """,
        
        """
        Deep Learning and Neural Networks
        
        Deep Learning is a subfield of machine learning that uses neural networks with 
        multiple layers to analyze various factors of data. These neural networks are 
        inspired by the structure and function of the human brain.
        
        Key concepts in deep learning include:
        - Artificial Neurons: Basic processing units that receive inputs and produce outputs
        - Layers: Collections of neurons that process information
        - Activation Functions: Mathematical functions that determine neuron output
        - Backpropagation: Algorithm for training neural networks
        
        Popular deep learning architectures include:
        - Convolutional Neural Networks (CNNs) for image processing
        - Recurrent Neural Networks (RNNs) for sequential data
        - Transformers for natural language processing
        """,
        
        """
        Ethics and Future of AI
        
        As AI technology advances, it raises important ethical considerations that need 
        to be addressed to ensure beneficial development and deployment.
        
        Key ethical concerns include:
        - Bias and Fairness: AI systems can perpetuate or amplify existing biases
        - Privacy: AI systems often require large amounts of personal data
        - Accountability: Who is responsible when AI systems make mistakes?
        - Job Displacement: AI automation may replace certain jobs
        - Safety: Ensuring AI systems operate safely and reliably
        
        The future of AI holds tremendous potential for transforming various industries 
        and improving human capabilities. Researchers are working on developing more 
        general AI systems that can perform a wide range of tasks, similar to human intelligence.
        """
    ]
    
    # Ingest documents
    rag_pipeline.ingest_documents(documents)
    print("✅ RAG server ready! Documents ingested and ready for queries.")

# Web interface
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Simple web interface for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Document Q&A System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            .query-box { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
            .response { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .source { background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }
            .timing { color: #666; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <h1>🤖 RAG Document Q&A System</h1>
        <div class="container">
            <h2>Ask a question about AI, Machine Learning, or Deep Learning:</h2>
            <input type="text" id="question" class="query-box" placeholder="What is artificial intelligence?">
            <button onclick="askQuestion()" class="btn">Ask Question</button>
            <div id="response"></div>
        </div>

        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                const responseDiv = document.getElementById('response');
                
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                responseDiv.innerHTML = '<div class="response">🔄 Processing your question...</div>';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const result = await response.json();
                    
                    let html = '<div class="response">';
                    html += '<h3>🤖 Answer:</h3>';
                    html += '<p>' + result.answer + '</p>';
                    html += '<h4>📋 Sources:</h4>';
                    
                    result.sources.forEach((source, index) => {
                        html += '<div class="source">';
                        html += '<strong>Source ' + (index + 1) + ':</strong> ' + source.metadata.source + ' (Relevance: ' + source.relevance_score.toFixed(3) + ')<br>';
                        html += '<em>' + source.content + '</em>';
                        html += '</div>';
                    });
                    
                    html += '<div class="timing">';
                    html += '⏱️ Retrieval: ' + result.timing.retrieval_time.toFixed(3) + 's | ';
                    html += 'LLM: ' + result.timing.llm_time.toFixed(3) + 's | ';
                    html += 'Total: ' + result.timing.total_time.toFixed(3) + 's';
                    html += '</div>';
                    html += '</div>';
                    
                    responseDiv.innerHTML = html;
                    
                } catch (error) {
                    responseDiv.innerHTML = '<div class="response">❌ Error: ' + error.message + '</div>';
                }
            }
            
            // Allow Enter key to submit
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG system is running"}

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG pipeline"""
    try:
        print(f"🔍 Received query: {request.question}")
        
        # Process query
        response = rag_pipeline.query(request.question)
        
        return QueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            question=response["question"],
            timing=response["timing"]
        )
        
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get stats
@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    return {
        "status": "healthy",
        "documents_processed": rag_pipeline.documents_processed,
        "system": "RAG Document Q&A System",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Server...")
    print("📱 Web Interface: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
