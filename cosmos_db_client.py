import os
import uuid
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

load_dotenv()

class CosmosDBClient:
    def __init__(self, database_name, container_name):
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        
        if not endpoint or not key:
            raise ValueError("Missing COSMOS_ENDPOINT or COSMOS_KEY environment variables")
        
        self.client = CosmosClient(endpoint, key)
        
        self.database = self._get_or_create_database(database_name)
        self.container = self._get_or_create_container(container_name)
        
        print(f"Connected to CosmosDB database '{database_name}', container '{container_name}'")
    
    def _get_or_create_database(self, database_name):
        return self.client.create_database_if_not_exists(database_name)
    
    def _get_or_create_container(self, container_name):
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
        
        return self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/sourceDocumentId"),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_policy,
            full_text_policy=fulltext_policy
        )
        
    def upsert_document_chunks(self, chunks, source_document_id=None):
        if not source_document_id:
            source_document_id = str(uuid.uuid4())
        
        uploaded_ids = []
        
        for i, chunk in enumerate(chunks):
            doc = {
                "id": f"{source_document_id}_chunk_{i}",
                "sourceDocumentId": source_document_id,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": chunk["embedding"],
                "chunkIndex": i,
                "totalChunks": len(chunks)
            }
            
            response = self.container.upsert_item(doc)
            uploaded_ids.append(response["id"])
            print(f"Uploaded chunk {i} with ID {response['id']} to CosmosDB")
            
        print(f"Uploaded {len(chunks)} chunks to CosmosDB for document {source_document_id}")
        return uploaded_ids
    
    def get_document_chunks(self, source_document_id):
        query = f"SELECT * FROM c WHERE c.sourceDocumentId = '{source_document_id}' ORDER BY c.chunkIndex"
        return list(self.container.query_items(query, enable_cross_partition_query=True))
    
    def delete_document_chunks(self, source_document_id):
        query = f"SELECT * FROM c WHERE c.sourceDocumentId = '{source_document_id}'"
        items = list(self.container.query_items(query, enable_cross_partition_query=True))
        
        for item in items:
            self.container.delete_item(item, partition_key=item["id"])
        
        print(f"Deleted {len(items)} chunks for document {source_document_id}")

    def full_text_search(self, query_text, top_k=5):
        # Split the query text into words for full-text search
        query_terms = query_text.split()
        query_terms_list = ', '.join([f'"{term}"' for term in query_terms])

        query = f"""
        SELECT TOP {top_k} 
            c.id, 
            c.content, 
            c.metadata, 
            c.sourceDocumentId, 
            c.chunkIndex
        FROM c ORDER BY RANK FullTextScore(c.content, [{query_terms_list}])
        """
        
        results = list(self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        # Add search score property to results
        for i, item in enumerate(results):
            # Since Cosmos SQL doesn't return search score directly, we simulate it
            item["searchScore"] = 1.0 - (i * (0.7 / len(results))) if results else 0
            
        return results

    def vector_search(self, embedding, top_k=5, min_similarity=0.7):
        query = f"""
        SELECT TOP {top_k} 
            c.id, 
            c.content, 
            c.metadata, 
            c.sourceDocumentId, 
            c.chunkIndex,
            VectorDistance(c.embedding, @embedding) AS searchScore 
        FROM c 
        WHERE VectorDistance(c.embedding, @embedding) > @minSimilarity 
        ORDER BY VectorDistance(c.embedding, @embedding)
        """

        parameters = [
            {"name": "@embedding", "value": embedding},
            {"name": "@minSimilarity", "value": min_similarity}
        ]
        
        results = list(self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        return results
        
    def hybrid_search(self, query_text, embedding, top_k=5):
        query_terms = query_text.split()
        query_terms_list = ', '.join([f'"{term}"' for term in query_terms])
        embedding_str = ', '.join(map(str, embedding))

        query = f"""
        SELECT TOP {top_k} 
            c.id, 
            c.content, 
            c.metadata, 
            c.sourceDocumentId, 
            c.chunkIndex
        FROM c ORDER BY RANK RRF(FullTextScore(c.content, [{query_terms_list}]), VectorDistance(c.embedding, [{embedding_str}]))
        """
        
        results = list(self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        # Add search score property to results
        for i, item in enumerate(results):
            # Since Cosmos SQL doesn't return search score directly, we simulate it
            item["searchScore"] = 1.0 - (i * (0.7 / len(results))) if results else 0
        
        return results
    
    def get_unique_document_sources(self):
        query = "SELECT DISTINCT VALUE c.sourceDocumentId FROM c"
        return list(self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
