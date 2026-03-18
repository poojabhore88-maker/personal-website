import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, chunking, and preprocessing for RAG pipeline."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitters
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Token-based splitter for more precise chunking
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base"  # OpenAI's encoding
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its file type."""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                logger.info(f"Loaded PDF: {file_path} with {len(documents)} pages")
                
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                logger.info(f"Loaded text file: {file_path}")
                
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Add metadata
            for doc in documents:
                doc.metadata['source'] = file_path
                doc.metadata['file_type'] = file_extension
                
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        try:
            # Use DirectoryLoader for multiple files
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.{pdf,txt,md}",
                loader_kwargs={'encoding': 'utf-8'},
                recursive=True,
                show_progress=True
            )
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from directory {directory_path}: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Document], strategy: str = "recursive") -> List[Document]:
        """Split documents into chunks using specified strategy."""
        try:
            if strategy == "recursive":
                chunks = self.char_splitter.split_documents(documents)
            elif strategy == "token":
                chunks = self.token_splitter.split_documents(documents)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)
                chunk.metadata['strategy'] = strategy
            
            logger.info(f"Created {len(chunks)} chunks using {strategy} strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            import io
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
            
            logger.info(f"Extracted text from PDF with {len(pdf_reader.pages)} pages")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        # (keeping basic punctuation)
        import re
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        return text.strip()
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the processed documents."""
        if not documents:
            return {"total_documents": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_chunks = len(documents)
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters for English)
        total_tokens = total_chars // 4
        
        file_types = {}
        sources = set()
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            sources.add(doc.metadata.get('source', 'unknown'))
        
        return {
            "total_documents": len(sources),
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "estimated_tokens": total_tokens,
            "file_types": file_types,
            "average_chunk_size": total_chars // total_chunks if total_chunks > 0 else 0
        }


# Example usage and testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with a sample document
    sample_text = """
    This is a sample document for testing the document processor.
    It contains multiple paragraphs and should be split into chunks.
    
    The RAG (Retrieval-Augmented Generation) system needs to process
    documents efficiently to provide context-aware responses.
    
    Document chunking is a crucial step in this process, as it determines
    how well the system can retrieve relevant information.
    """
    
    # Create a test document
    test_doc = Document(page_content=sample_text, metadata={"source": "test"})
    
    # Test chunking
    chunks = processor.chunk_documents([test_doc])
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk.page_content)} characters")
