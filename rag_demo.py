#!/usr/bin/env python3
"""
Simple RAG Demo - Shows the output of the RAG agent without requiring server setup
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from document_processor import DocumentProcessor
    from vector_store import VectorStore
    from rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

def create_sample_documents():
    """Create sample documents for demonstration."""
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
    
    return documents

def demonstrate_rag_pipeline():
    """Demonstrate the complete RAG pipeline functionality."""
    print("🚀 RAG Agent Demonstration")
    print("=" * 60)
    
    # Create sample documents
    print("\n📄 Creating sample documents...")
    documents = create_sample_documents()
    print(f"✅ Created {len(documents)} sample documents")
    
    # Initialize document processor
    print("\n🔧 Initializing document processor...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    # Create Document objects
    from langchain.docstore.document import Document
    doc_objects = []
    for i, content in enumerate(documents):
        doc = Document(
            page_content=content,
            metadata={"source": f"sample_doc_{i+1}.txt", "doc_id": i+1}
        )
        doc_objects.append(doc)
    
    # Chunk documents
    print("📝 Chunking documents...")
    chunks = processor.chunk_documents(doc_objects, strategy="recursive")
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show sample chunks
    print("\n📋 Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Content: {chunk.page_content[:150]}...")
        print(f"Size: {len(chunk.page_content)} characters")
    
    # Initialize vector store (using Hugging Face embeddings for demo)
    print("\n🔍 Initializing vector store...")
    try:
        vector_store = VectorStore()
        vector_store.create_vector_store(chunks)
        print("✅ Vector store created successfully")
        
        # Show vector store stats
        stats = vector_store.get_vector_store_stats()
        print(f"📊 Vector store stats:")
        print(f"   - Total vectors: {stats.get('total_vectors', 'N/A')}")
        print(f"   - Dimension: {stats.get('dimension', 'N/A')}")
        print(f"   - Index type: {stats.get('index_type', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        print("Note: This requires embedding models to be installed")
        return
    
    # Test similarity search
    print("\n🔍 Testing similarity search...")
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the ethical concerns with AI?",
        "What are neural networks?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        try:
            results = vector_store.similarity_search(query, k=2)
            print(f"✅ Found {len(results)} relevant documents:")
            
            for i, doc in enumerate(results, 1):
                print(f"\n   Result {i}:")
                print(f"   Source: {doc.metadata['source']}")
                print(f"   Content: {doc.page_content[:200]}...")
                
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    # Demonstrate RAG responses (mock since we don't have LLM API keys)
    print("\n🤖 Simulating RAG Responses:")
    print("=" * 40)
    
    mock_responses = {
        "What is artificial intelligence?": "Artificial Intelligence is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, problem-solving, perception, and language understanding. The field was founded in 1956 and has evolved through various approaches including machine learning and deep learning.",
        
        "How does machine learning work?": "Machine Learning enables systems to learn and improve from experience without being explicitly programmed. ML algorithms use training data to make predictions or decisions. The three main types are supervised learning (with labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error). Applications include recommendation systems, spam filtering, and medical diagnosis.",
        
        "What are the ethical concerns with AI?": "Key ethical concerns in AI include bias and fairness (AI systems can perpetuate existing biases), privacy (requiring large amounts of personal data), accountability (who is responsible for AI mistakes), job displacement (automation replacing jobs), and safety (ensuring reliable operation). These issues need to be addressed for beneficial AI development.",
        
        "What are neural networks?": "Neural networks are computing systems inspired by the human brain, consisting of interconnected artificial neurons. Key components include artificial neurons (basic processing units), layers (collections of neurons), activation functions (determine neuron output), and backpropagation (training algorithm). Popular architectures include CNNs for images, RNNs for sequential data, and Transformers for NLP."
    }
    
    for query, response in mock_responses.items():
        print(f"\n📝 Question: {query}")
        print(f"🤖 RAG Response: {response}")
        print(f"📊 Sources: Retrieved from {len(chunks)} document chunks")
        print(f"⚡ Processing time: ~0.5s (simulated)")
        print("-" * 50)
    
    print("\n✨ RAG Agent Demo Complete!")
    print("\n📚 Key Features Demonstrated:")
    print("   ✅ Document processing and chunking")
    print("   ✅ Vector similarity search")
    print("   ✅ Context retrieval")
    print("   ✅ Context-aware responses")
    print("   ✅ Source attribution")
    
    print("\n🚀 To run the full system:")
    print("   1. Set up API keys in .env file")
    print("   2. Install dependencies: pip install -r requirements.txt")
    print("   3. Start server: python main.py")
    print("   4. Test: python test_client.py")

if __name__ == "__main__":
    demonstrate_rag_pipeline()
