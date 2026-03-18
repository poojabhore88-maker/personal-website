import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Model Configuration
    embedding_model: str = "text-embedding-ada-002"
    huggingface_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chat_model: str = "gpt-3.5-turbo"
    
    # FAISS Configuration
    faiss_index_path: str = "./faiss_index"
    
    # FastAPI Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4000
    
    # RAG Configuration
    retrieval_k: int = 4
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
