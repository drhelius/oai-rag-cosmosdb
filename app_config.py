"""
Configuration settings for the Streamlit RAG demo app.
"""

# Document processing defaults
DEFAULT_SETTINGS = {
    # Blob Storage settings
    "blob_container_name": "docs",
    
    # CosmosDB settings
    "cosmos_db_name": "tr",
    "cosmos_container_name": "embeddings",
    
    # Text splitting settings
    "tokens_per_chunk": 512,
    "overlap_tokens": 128,
    
    # Embedding model
    "embedding_model": "text_embedding_3_small",
    
    # LLM model for chat
    "llm_model": "gpt4o_1"
}

# UI settings
UI_CONFIG = {
    "max_file_size_mb": 50,
    "allowed_file_types": ["pdf"],
    "max_files": 10,
    "temp_upload_dir": "temp_uploads"
}

# Demo metrics labels
METRICS = {
    "document_count": "Documents indexed",
    "total_pages": "Total pages processed",
    "total_chunks": "Text chunks created",
    "cosmos_db_items": "CosmosDB items",
    "processing_time": "Processing time (sec)",
    "avg_embedding_time": "Avg. embedding time per chunk (sec)"
}

# Search configuration
SEARCH_CONFIG = {
    "top_k": 5,
    "min_similarity": 0.7,
    "embedding_model": "text_embedding_3_small"
}

# Chat configuration
CHAT_CONFIG = {
    "llm_model": "gpt4o_1",
    "embedding_model": "text_embedding_3_small",
    "search_type": "hybrid",
    "top_k": 3,
    "min_similarity": 0.7,
    "temperature": 0.7,
    "system_prompt": """You are an AI assistant helping with questions about documents stored in a database.
For each user question, relevant document excerpts will be provided as context.
Base your answers primarily on this context. If the context doesn't contain the answer, say so clearly.
Always cite your sources by mentioning the Document ID when you reference information from the context. Give the user the URL of the document if available.
Keep your answers clear, helpful, and accurate."""
}
