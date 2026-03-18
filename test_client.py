#!/usr/bin/env python3
"""
Test client for the RAG Document Q&A System.
This script demonstrates how to interact with the API endpoints.
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_DOCS = f"{BASE_URL}/docs"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def create_sample_document():
    """Create a sample document for testing."""
    sample_content = """
    Artificial Intelligence and Machine Learning

    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that can perform tasks that typically require human intelligence. These tasks include learning, 
    reasoning, problem-solving, perception, and language understanding.

    Machine Learning (ML) is a subset of AI that enables systems to learn and improve from experience 
    without being explicitly programmed. ML algorithms use training data to make predictions or decisions.

    Deep Learning is a subfield of machine learning that uses neural networks with multiple layers 
    to analyze various factors of data. These neural networks are inspired by the structure and 
    function of the human brain.

    Applications of AI and ML include:
    - Natural Language Processing (NLP)
    - Computer Vision
    - Speech Recognition
    - Recommendation Systems
    - Autonomous Vehicles
    - Healthcare Diagnostics
    - Financial Trading

    The future of AI holds tremendous potential for transforming various industries and improving 
    human capabilities. However, it also raises important ethical considerations that need to be 
    addressed as the technology continues to evolve.
    """
    
    # Create sample file
    with open("sample_ai_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    return "sample_ai_document.txt"

def test_document_ingestion():
    """Test document ingestion."""
    print("\n📄 Testing document ingestion...")
    
    # Create sample document
    sample_file = create_sample_document()
    
    try:
        # Upload the sample document
        with open(sample_file, "rb") as f:
            files = {"files": f}
            data = {
                "chunk_size": 500,
                "chunk_overlap": 100
            }
            
            response = requests.post(f"{BASE_URL}/ingest", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document ingestion successful:")
            print(f"   - Documents processed: {result.get('documents_processed', 0)}")
            print(f"   - Chunks created: {result.get('chunks_created', 0)}")
            print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"❌ Document ingestion failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Document ingestion error: {e}")
        return False
    finally:
        # Clean up sample file
        if os.path.exists(sample_file):
            os.remove(sample_file)

def test_query():
    """Test the query endpoint."""
    print("\n🤖 Testing query functionality...")
    
    test_queries = [
        "What is Artificial Intelligence?",
        "How does Machine Learning work?",
        "What are the applications of AI?",
        "What is Deep Learning?",
        "What are the ethical considerations of AI?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        try:
            query_data = {
                "question": query,
                "search_type": "similarity",
                "k": 3
            }
            
            response = requests.post(f"{BASE_URL}/query", json=query_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response received:")
                print(f"   - Answer: {result['answer'][:200]}...")
                print(f"   - Sources: {len(result['sources'])}")
                print(f"   - Response time: {result['response_time']:.2f}s")
                
                # Show source relevance scores
                if result['relevance_scores']:
                    print(f"   - Relevance scores: {[f'{score:.3f}' for score in result['relevance_scores'][:3]]}")
            else:
                print(f"❌ Query failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Query error: {e}")
        
        time.sleep(1)  # Rate limiting

def test_retrieve():
    """Test the retrieve endpoint (without LLM)."""
    print("\n🔍 Testing document retrieval...")
    
    try:
        data = {
            "question": "What are the applications of AI?",
            "k": 3
        }
        
        response = requests.post(f"{BASE_URL}/retrieve", data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieval successful:")
            print(f"   - Documents found: {result['count']}")
            
            for i, doc in enumerate(result['documents'][:2], 1):
                print(f"   - Document {i}: {doc['content'][:100]}...")
                
        else:
            print(f"❌ Retrieval failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Retrieval error: {e}")

def test_stats():
    """Test the statistics endpoint."""
    print("\n📊 Testing pipeline statistics...")
    
    try:
        response = requests.get(f"{BASE_URL}/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Statistics retrieved:")
            print(f"   - LLM Model: {stats.get('llm_model', 'N/A')}")
            print(f"   - Embedding Model: {stats.get('embedding_model', 'N/A')}")
            print(f"   - Retrieval K: {stats.get('retrieval_k', 'N/A')}")
            print(f"   - Temperature: {stats.get('temperature', 'N/A')}")
            
            vector_store = stats.get('vector_store', {})
            print(f"   - Vector Store Status: {vector_store.get('status', 'N/A')}")
            if vector_store.get('total_vectors'):
                print(f"   - Total Vectors: {vector_store.get('total_vectors', 'N/A')}")
                
        else:
            print(f"❌ Stats retrieval failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Stats error: {e}")

def test_different_search_types():
    """Test different search strategies."""
    print("\n🔄 Testing different search types...")
    
    query = "What is Machine Learning?"
    search_types = ["similarity", "mmr"]
    
    for search_type in search_types:
        print(f"\n🔍 Testing {search_type} search...")
        
        try:
            query_data = {
                "question": query,
                "search_type": search_type,
                "k": 3
            }
            
            response = requests.post(f"{BASE_URL}/query", json=query_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {search_type.title()} search successful:")
                print(f"   - Answer length: {len(result['answer'])} characters")
                print(f"   - Sources: {len(result['sources'])}")
                print(f"   - Response time: {result['response_time']:.2f}s")
            else:
                print(f"❌ {search_type.title()} search failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {search_type.title()} search error: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting RAG System Tests")
    print("=" * 50)
    
    # Check if server is running
    if not test_health_check():
        print("\n❌ Server is not running. Please start the server first:")
        print("   python main.py")
        print("   or")
        print("   docker-compose up")
        return
    
    # Run tests
    test_document_ingestion()
    test_query()
    test_retrieve()
    test_stats()
    test_different_search_types()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print(f"📚 API Documentation available at: {API_DOCS}")

if __name__ == "__main__":
    main()
