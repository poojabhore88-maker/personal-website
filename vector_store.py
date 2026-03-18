import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages FAISS vector store for document retrieval."""
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or settings.embedding_model
        self.faiss_index_path = settings.faiss_index_path
        self.embeddings = self._initialize_embeddings()
        self.vector_store: Optional[FAISS] = None
        
        # Create directory if it doesn't exist
        Path(self.faiss_index_path).mkdir(parents=True, exist_ok=True)
    
    def _initialize_embeddings(self):
        """Initialize the embedding model based on configuration."""
        try:
            if settings.openai_api_key:
                logger.info("Using OpenAI embeddings")
                return OpenAIEmbeddings(
                    model=self.embedding_model,
                    openai_api_key=settings.openai_api_key
                )
            else:
                logger.info("Using Hugging Face embeddings")
                return HuggingFaceEmbeddings(
                    model_name=settings.huggingface_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents."""
        try:
            logger.info(f"Creating vector store from {len(documents)} documents")
            
            # Create FAISS index
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save the vector store
            self.save_vector_store()
            
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self) -> bool:
        """Load existing vector store from disk."""
        try:
            index_path = os.path.join(self.faiss_index_path, "index.faiss")
            docstore_path = os.path.join(self.faiss_index_path, "index.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(docstore_path):
                logger.warning("Vector store files not found")
                return False
            
            self.vector_store = FAISS.load_local(
                self.faiss_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def save_vector_store(self):
        """Save vector store to disk."""
        try:
            if self.vector_store is None:
                raise ValueError("No vector store to save")
            
            self.vector_store.save_local(self.faiss_index_path)
            logger.info(f"Vector store saved to {self.faiss_index_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the existing vector store."""
        try:
            if self.vector_store is None:
                self.create_vector_store(documents)
            else:
                self.vector_store.add_documents(documents)
                self.save_vector_store()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search for query."""
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            
            k = k or settings.retrieval_k
            results = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def similarity_search_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores."""
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            
            k = k or settings.retrieval_k
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {str(e)}")
            raise
    
    def max_marginal_relevance_search(self, query: str, k: int = None, fetch_k: int = 20) -> List[Document]:
        """Perform Max Marginal Relevance search for diverse results."""
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            
            k = k or settings.retrieval_k
            results = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k
            )
            
            logger.info(f"Found {len(results)} diverse documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in MMR search: {str(e)}")
            raise
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            if self.vector_store is None:
                return {"status": "not_initialized"}
            
            # Get index information
            index = self.vector_store.index
            docstore = self.vector_store.docstore
            
            stats = {
                "status": "initialized",
                "index_type": type(index).__name__,
                "total_vectors": index.ntotal,
                "dimension": index.d if hasattr(index, 'd') else "unknown",
                "docstore_count": len(docstore._dict),
                "embedding_model": self.embedding_model,
                "index_path": self.faiss_index_path
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def delete_vector_store(self):
        """Delete the vector store from disk."""
        try:
            import shutil
            if os.path.exists(self.faiss_index_path):
                shutil.rmtree(self.faiss_index_path)
                logger.info(f"Deleted vector store at {self.faiss_index_path}")
            
            self.vector_store = None
            
        except Exception as e:
            logger.error(f"Error deleting vector store: {str(e)}")
            raise
    
    def rebuild_vector_store(self, documents: List[Document]):
        """Rebuild the entire vector store."""
        try:
            logger.info("Rebuilding vector store...")
            self.delete_vector_store()
            self.create_vector_store(documents)
            logger.info("Vector store rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding vector store: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Example documents
    documents = [
        Document(
            page_content="Artificial Intelligence is a branch of computer science that aims to create intelligent machines.",
            metadata={"source": "ai_intro.txt"}
        ),
        Document(
            page_content="Machine Learning is a subset of AI that enables systems to learn and improve from experience.",
            metadata={"source": "ml_intro.txt"}
        ),
        Document(
            page_content="Deep Learning uses neural networks with multiple layers to analyze various factors of data.",
            metadata={"source": "dl_intro.txt"}
        )
    ]
    
    # Initialize vector store
    vs = VectorStore()
    
    # Create and test
    vs.create_vector_store(documents)
    
    # Test search
    results = vs.similarity_search("What is artificial intelligence?")
    print(f"Search results: {len(results)} documents found")
    
    # Print stats
    stats = vs.get_vector_store_stats()
    print(f"Vector store stats: {stats}")
