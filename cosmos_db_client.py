import os
import uuid
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

load_dotenv()

class CosmosDBClient:
    def __init__(self, database_name, container_name):
        """
        Initialize the CosmosDB client.
        
        Args:
            database_name (str): Name of the Cosmos DB database
            container_name (str): Name of the container within the database
        """
        # Get connection parameters from environment variables
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        
        if not endpoint or not key:
            raise ValueError("Missing COSMOS_ENDPOINT or COSMOS_KEY environment variables")
        
        # Initialize the Cosmos client
        self.client = CosmosClient(endpoint, key)
        
        # Create database and container if they don't exist
        self.database = self._get_or_create_database(database_name)
        self.container = self._get_or_create_container(container_name)
        
        print(f"Connected to CosmosDB database '{database_name}', container '{container_name}'")
    
    def _get_or_create_database(self, database_name):
        """Create database if it doesn't exist and return the database client."""
        return self.client.create_database_if_not_exists(database_name)
    
    def _get_or_create_container(self, container_name):
        """Create container if it doesn't exist with required policies and return the container client."""

        # Define indexing policy with vector and full-text search support
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [
                {
                    "path": "/*"
                }
            ],
            "excludedPaths": [
                {
                    "path": "/\"_etag\"/?"
                },
                {
                    "path": "/embedding/*"
                }
            ],
            "fullTextIndexes": [
                {
                    "path": "/content"
                }
            ],
            "vectorIndexes": [
                {
                    "path": "/embedding",
                    "type": "diskANN"
                }
            ]
        }
        
        # Define vector embeddings policy
        vector_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 1536
                }
            ]
        }

        # Define full-text search policy
        fulltext_policy = {
            "defaultLanguage": "en-US",
            "fullTextPaths": [
                {
                    "path": "/content",
                    "language": "en-US"
                }
            ]
        }
        
        # Create container with the specified policies
        return self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/sourceDocumentId"),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_policy,
            full_text_policy=fulltext_policy
        )

        
    def upsert_document_chunks(self, chunks, source_document_id=None):
        """
        Upload document chunks with embeddings to Cosmos DB.
        
        Args:
            chunks (list): List of chunk dictionaries, each containing 'content', 'metadata', and 'embedding'
            source_document_id (str): Optional identifier to group chunks from the same document
            
        Returns:
            list: List of created document IDs
        """
        # Generate a document ID if not provided
        if not source_document_id:
            source_document_id = str(uuid.uuid4())
        
        uploaded_ids = []
        
        for i, chunk in enumerate(chunks):
            # Create a document for this chunk
            doc = {
                "id": f"{source_document_id}_chunk_{i}",
                "sourceDocumentId": source_document_id,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": chunk["embedding"],
                "chunkIndex": i,
                "totalChunks": len(chunks)
            }
            
            # Upload to CosmosDB
            response = self.container.upsert_item(doc)
            uploaded_ids.append(response["id"])
            print(f"Uploaded chunk {i} with ID {response['id']} to CosmosDB")
            
        print(f"Uploaded {len(chunks)} chunks to CosmosDB for document {source_document_id}")
        return uploaded_ids
    
    def get_document_chunks(self, source_document_id):
        """
        Retrieve all chunks for a specific document.
        
        Args:
            source_document_id (str): The source document identifier
            
        Returns:
            list: List of document chunks
        """
        query = f"SELECT * FROM c WHERE c.sourceDocumentId = '{source_document_id}' ORDER BY c.chunkIndex"
        return list(self.container.query_items(query, enable_cross_partition_query=True))
    
    def delete_document_chunks(self, source_document_id):
        """
        Delete all chunks for a specific document.
        
        Args:
            source_document_id (str): The source document identifier
        """
        query = f"SELECT * FROM c WHERE c.sourceDocumentId = '{source_document_id}'"
        items = list(self.container.query_items(query, enable_cross_partition_query=True))
        
        for item in items:
            self.container.delete_item(item, partition_key=item["id"])
        
        print(f"Deleted {len(items)} chunks for document {source_document_id}")
