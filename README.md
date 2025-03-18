# Azure CosmosDB RAG Demo Application

This repository contains a Retrieval-Augmented Generation (RAG) demo application built with Azure CosmosDB, Azure OpenAI Service, and Streamlit. The application demonstrates how to implement different search mechanisms (full-text, vector, and hybrid search) using CosmosDB's capabilities.

## Architecture

This application implements a RAG pattern with the following components:

1. **Document Processing Pipeline**:
   - PDF document ingestion and text extraction with PyMuPDF4LLM
   - Text chunking with tiktoken
   - Vector embedding generation using Azure OpenAI embeddings models
   - Storage in Azure Blob Storage (original files) and CosmosDB (chunks + embeddings)

2. **Search Service**:
   - Full-text search using CosmosDB's built-in capabilities
   - Vector similarity search using CosmosDB's vector store
   - Hybrid search combining both approaches with reciprocal rank fusion

3. **Chat Service**:
   - Integration with Azure OpenAI's models
   - Context retrieval from indexed documents
   - Streaming response generation

## Setup Instructions

### Prerequisites

- Azure subscription with:
  - Azure CosmosDB account with SQL API
  - Azure OpenAI Service resources
  - Azure Blob Storage account
- Python 3.8+

### Environment Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/oai-rag-cosmosdb.git
cd oai-rag-cosmosdb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables by copying the `.env.example` file (if available) to `.env` and filling in your Azure service details:
```bash
cp .env.example .env
```

4. Edit the `.env` file with your Azure service details:
   - OpenAI API keys and endpoints
   - Azure Blob Storage connection string
   - CosmosDB endpoint and key

### Running the Application

Launch the application with:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage Guide

### Document Loading & Indexing

1. Navigate to the "Document Loading & Indexing" tab
2. Upload PDF documents (up to 50MB each)
3. Configure chunk size and overlap settings if needed
4. Click "Process Documents" to extract text, generate embeddings, and store in CosmosDB
5. View processing metrics and history

### Document Search

1. Navigate to the "Document Search" tab
2. Enter your search query
3. Select a search method (full-text, vector, or hybrid search)
4. Configure search parameters like top-k results and minimum similarity score
5. Toggle "Compare Search Methods" to see performance differences between methods
6. Review search results and performance metrics

### Chat with Documents

1. Navigate to the "LLM Chat with RAG" tab
2. Configure your chat settings (search type, number of context chunks, LLM model)
3. Ask questions about your documents
4. Review AI-generated responses with citations to source documents
5. Optionally view the retrieved context used to generate responses
6. Track chat performance metrics

