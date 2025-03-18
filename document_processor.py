import os
import time
import streamlit as st
from blob_storage_client import BlobStorageClient
from pdf_converter import PdfConverter
from text_splitter import TextSplitter
from embedding_generator import EmbeddingGenerator
from cosmos_db_client import CosmosDBClient
from app_utils import generate_document_id

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.blob_client = BlobStorageClient(config["blob_container_name"])
        self.cosmos_db_client = CosmosDBClient(config["cosmos_db_name"], config["cosmos_container_name"])
        self.splitter = TextSplitter(tokens_per_chunk=config["tokens_per_chunk"], 
                                     overlap_tokens=config["overlap_tokens"])
        self.embedding_generator = EmbeddingGenerator(model_id=config["embedding_model"])
        
        self.metrics = {
            "document_count": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "cosmos_db_items": 0,
            "processing_time": 0,
            "avg_embedding_time": 0,
            "document_details": []
        }
    
    def process_document(self, file_path, progress_bar=None, status_text=None):
        start_time = time.time()
        doc_metrics = {"filename": os.path.basename(file_path)}
        
        if status_text:
            status_text.text("1/5: Uploading to Blob Storage...")
        
        # Upload original PDF to blob storage
        blob_name = os.path.basename(file_path)
        self.blob_client.upload_file(local_file_path=file_path)
        blob_url = self.blob_client.get_blob_url(blob_name)
        doc_metrics["blob_url"] = blob_url
        
        if progress_bar:
            progress_bar.progress(20)
        if status_text:
            status_text.text("2/5: Converting PDF to text...")
        
        # Convert PDF to Markdown and get pages data
        pages = PdfConverter.pdf_to_markdown(file_path)
        doc_metrics["pages"] = len(pages)
        self.metrics["total_pages"] += len(pages)
        
        if progress_bar:
            progress_bar.progress(40)
        if status_text:
            status_text.text("3/5: Splitting into chunks...")
        
        # Split markdown content into chunks
        chunks = self.splitter.split_pages(pages)
        doc_metrics["chunks"] = len(chunks)
        self.metrics["total_chunks"] += len(chunks)
        
        if progress_bar:
            progress_bar.progress(60)
        if status_text:
            status_text.text("4/5: Generating embeddings...")
        
        # Generate embeddings for chunks
        embedding_start_time = time.time()
        embeddings = self.embedding_generator.generate_embeddings_for_chunks([chunk["content"] for chunk in chunks])
        embedding_time = time.time() - embedding_start_time
        doc_metrics["embedding_time"] = embedding_time
        
        # Calculate average embedding time per chunk
        if len(chunks) > 0:
            chunk_avg_time = embedding_time / len(chunks)
            doc_metrics["avg_chunk_time"] = chunk_avg_time
            
            # Update overall average (weighted)
            total_chunks = self.metrics["total_chunks"]
            if total_chunks > 0:
                prev_avg = self.metrics.get("avg_embedding_time", 0)
                prev_chunks = total_chunks - len(chunks)
                new_avg = (prev_avg * prev_chunks + chunk_avg_time * len(chunks)) / total_chunks
                self.metrics["avg_embedding_time"] = new_avg
        
        if progress_bar:
            progress_bar.progress(80)
        if status_text:
            status_text.text("5/5: Storing in CosmosDB...")
        
        # Combine chunks with their embeddings and add metadata
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk["embedding"] = embedding
            chunk["metadata"]["source_url"] = blob_url
        
        # Upload chunks to CosmosDB
        document_id = generate_document_id(file_path)
        doc_metrics["document_id"] = document_id
        cosmos_ids = self.cosmos_db_client.upsert_document_chunks(chunks, source_document_id=document_id)
        doc_metrics["cosmos_items"] = len(cosmos_ids)
        self.metrics["cosmos_db_items"] += len(cosmos_ids)
        
        if progress_bar:
            progress_bar.progress(100)
        if status_text:
            status_text.text("✅ Processing complete!")
        
        # Record processing time
        processing_time = time.time() - start_time
        doc_metrics["processing_time"] = processing_time
        self.metrics["processing_time"] += processing_time
        
        # Update document count
        self.metrics["document_count"] += 1
        
        # Store document details
        self.metrics["document_details"].append(doc_metrics)
        
        return document_id, doc_metrics
