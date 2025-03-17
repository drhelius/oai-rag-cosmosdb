import os
from blob_storage_client import BlobStorageClient
from pdf_converter import PdfConverter
from text_splitter import TextSplitter
from embedding_generator import EmbeddingGenerator
from cosmos_db_client import CosmosDBClient

if __name__ == "__main__":
    # Define configuration
    blob_container_name = "docs"
    cosmos_db_name = "tr"
    cosmos_container_name = "embeddings"
    
    # Initialize clients
    blob_client = BlobStorageClient(blob_container_name)
    cosmos_db_client = CosmosDBClient(cosmos_db_name, cosmos_container_name)

    # Document processing
    local_pdf_path = "docs/file.pdf"
    blob_name = os.path.basename(local_pdf_path)
    document_id = blob_name.replace('.pdf', '')  # Use filename as document ID

    # Upload original PDF to blob storage
    blob_client.upload_file(local_file_path=local_pdf_path)
    blob_url = blob_client.get_blob_url(blob_name)
    print(f"Uploaded PDF to {blob_url}")

    # Convert PDF to Markdown and get pages data
    pages = PdfConverter.pdf_to_markdown(local_pdf_path)

    # Split markdown content into chunks with metadata using page data directly
    splitter = TextSplitter(tokens_per_chunk=512, overlap_tokens=128)
    chunks = splitter.split_pages(pages)
    print(f"Created {len(chunks)} chunks from document")

    # Generate embeddings for chunks using Azure OpenAI
    embedding_generator = EmbeddingGenerator(model_id="text_embedding_3_small")
    embeddings = embedding_generator.generate_embeddings_for_chunks([chunk["content"] for chunk in chunks])
    
    # Combine chunks with their embeddings
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk["embedding"] = embedding
        # Add the blob URL to chunk metadata
        chunk["metadata"]["source_url"] = blob_url
        
    # Upload chunks to CosmosDB
    cosmos_db_client.upsert_document_chunks(chunks, source_document_id=document_id)
    
    print(f"Document processing complete. Document ID: {document_id}")



