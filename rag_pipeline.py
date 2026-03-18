import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

from document_processor import DocumentProcessor
from vector_store import VectorStore
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response structure for RAG pipeline."""
    answer: str
    source_documents: List[Document]
    relevance_scores: List[float]
    query: str
    response_time: float


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for document Q&A."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.vector_store = VectorStore()
        self.llm = self._initialize_llm()
        self.qa_chain = None
        self._initialize_qa_chain()
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        try:
            if settings.openai_api_key:
                logger.info("Using OpenAI LLM")
                return ChatOpenAI(
                    model=settings.chat_model,
                    openai_api_key=settings.openai_api_key,
                    temperature=settings.temperature
                )
            else:
                logger.info("Using Hugging Face LLM")
                return HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    huggingfacehub_api_token=settings.huggingface_api_key,
                    model_kwargs={"temperature": settings.temperature}
                )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def _initialize_qa_chain(self):
        """Initialize the QA chain with custom prompt."""
        try:
            # Custom prompt template for better responses
            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer from the context, just say that you don't know. 
            Try to provide a detailed and helpful answer based on the given context.

            Context:
            {context}

            Question: {question}

            Helpful Answer:"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            # Load existing vector store if available
            if not self.vector_store.load_vector_store():
                logger.warning("No existing vector store found. Please ingest documents first.")
                return

            # Create retriever
            retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.retrieval_k}
            )

            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            logger.info("QA chain initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            raise
    
    def ingest_documents(self, file_paths: List[str] = None, directory_path: str = None) -> Dict[str, Any]:
        """Ingest documents into the RAG pipeline."""
        try:
            import time
            start_time = time.time()

            documents = []

            # Load documents
            if file_paths:
                for file_path in file_paths:
                    docs = self.document_processor.load_document(file_path)
                    documents.extend(docs)
            
            elif directory_path:
                documents = self.document_processor.load_documents_from_directory(directory_path)
            
            else:
                raise ValueError("Either file_paths or directory_path must be provided")

            if not documents:
                raise ValueError("No documents found to ingest")

            # Chunk documents
            chunks = self.document_processor.chunk_documents(documents)
            
            # Create/update vector store
            self.vector_store.create_vector_store(chunks)
            
            # Reinitialize QA chain with new documents
            self._initialize_qa_chain()

            processing_time = time.time() - start_time
            stats = self.document_processor.get_document_stats(chunks)

            result = {
                "status": "success",
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "processing_time": processing_time,
                "stats": stats
            }

            logger.info(f"Successfully ingested {len(documents)} documents")
            return result

        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def query(self, question: str, search_type: str = "similarity") -> RAGResponse:
        """Query the RAG pipeline with a question."""
        try:
            import time
            start_time = time.time()

            if not self.qa_chain:
                raise ValueError("QA chain not initialized. Please ingest documents first.")

            # Perform different types of search based on search_type
            if search_type == "similarity":
                source_docs = self.vector_store.similarity_search(question)
            elif search_type == "similarity_with_scores":
                results = self.vector_store.similarity_search_with_scores(question)
                source_docs = [doc for doc, score in results]
                relevance_scores = [score for doc, score in results]
            elif search_type == "mmr":
                source_docs = self.vector_store.max_marginal_relevance_search(question)
                relevance_scores = [1.0] * len(source_docs)  # MMR doesn't provide scores
            else:
                raise ValueError(f"Unknown search type: {search_type}")

            # Generate answer
            result = self.qa_chain({"query": question})
            
            response_time = time.time() - start_time

            # If we didn't get relevance scores, calculate them
            if search_type == "similarity" or search_type == "mmr":
                results = self.vector_store.similarity_search_with_scores(question, k=len(source_docs))
                relevance_scores = [score for doc, score in results]

            response = RAGResponse(
                answer=result["result"],
                source_documents=source_docs,
                relevance_scores=relevance_scores,
                query=question,
                response_time=response_time
            )

            logger.info(f"Query processed in {response_time:.2f} seconds")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def get_relevant_documents(self, query: str, k: int = None) -> List[Document]:
        """Get relevant documents without generating an answer."""
        try:
            if not self.vector_store.vector_store:
                raise ValueError("Vector store not initialized")
            
            k = k or settings.retrieval_k
            documents = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Retrieved {len(documents)} relevant documents")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline."""
        try:
            stats = {
                "llm_model": settings.chat_model,
                "embedding_model": settings.embedding_model,
                "retrieval_k": settings.retrieval_k,
                "temperature": settings.temperature,
                "vector_store": self.vector_store.get_vector_store_stats()
            }
            
            return stats

        except Exception as e:
            logger.error(f"Error getting pipeline stats: {str(e)}")
            return {"error": str(e)}
    
    def update_settings(self, **kwargs):
        """Update pipeline settings."""
        try:
            for key, value in kwargs.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
                    logger.info(f"Updated {key} to {value}")
            
            # Reinitialize components if necessary
            if "openai_api_key" in kwargs or "huggingface_api_key" in kwargs:
                self.llm = self._initialize_llm()
                self._initialize_qa_chain()
            
            if "embedding_model" in kwargs or "huggingface_model" in kwargs:
                self.vector_store = VectorStore()
                self._initialize_qa_chain()

        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Example query (would need documents first)
    try:
        response = rag.query("What is artificial intelligence?")
        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.source_documents)} documents")
    except Exception as e:
        print(f"Error: {e}")
    
    # Get pipeline stats
    stats = rag.get_pipeline_stats()
    print(f"Pipeline stats: {stats}")
