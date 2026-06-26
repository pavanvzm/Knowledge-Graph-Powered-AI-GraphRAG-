"""Application configuration and settings management."""

import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Neo4j Configuration (Optional)
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # Vector Store Configuration
    vector_store_type: str = "in_memory"
    embedding_dimension: int = 1536
    
    # GraphRAG Configuration
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    max_hop_depth: int = 2
    
    # Data Directory
    data_dir: str = "data"
    
    @property
    def use_neo4j(self) -> bool:
        """Check if Neo4j is configured and should be used."""
        return all([
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password
        ])
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key, raising error if not configured."""
        if not self.has_openai_key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY in environment or .env file."
            )
        return self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
