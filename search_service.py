import time
from embedding_generator import EmbeddingGenerator

class SearchService:
    def __init__(self, cosmos_db_client, embedding_model="text_embedding_3_small"):
        """
        Service for performing different types of searches against CosmosDB.
        
        Args:
            cosmos_db_client: CosmosDBClient instance
            embedding_model: Model to use for embedding generation
        """
        self.cosmos_db = cosmos_db_client
        self.embedding_generator = EmbeddingGenerator(model_id=embedding_model)
        
    def text_search(self, query_text, top_k=5, min_similarity=0):
        """Perform full-text search in CosmosDB."""
        start_time = time.time()
        results = self.cosmos_db.full_text_search(query_text, top_k)
        search_time = time.time() - start_time
        
        search_metrics = {
            "search_type": "Full-Text Search",
            "query_time_ms": search_time * 1000,
            "results_count": len(results)
        }
        
        return results, search_metrics
        
    def vector_search(self, query_text, top_k=5, min_similarity=0.7):
        """Perform vector similarity search in CosmosDB."""
        # Generate embedding for query text
        start_time = time.time()
        embedding = self.embedding_generator.generate_embedding(query_text)
        
        # Perform vector search
        results = self.cosmos_db.vector_search(embedding, top_k, min_similarity)
        search_time = time.time() - start_time
        
        search_metrics = {
            "search_type": "Vector Search",
            "query_time_ms": search_time * 1000,
            "results_count": len(results),
            "embedding_dimensions": len(embedding) if embedding else 0
        }
        
        return results, search_metrics
    
    def hybrid_search(self, query_text, top_k=5, min_similarity=0.7, weights=None):
        """Perform hybrid search (combination of vector and text) in CosmosDB."""
        # Default weights if not provided
        if weights is None:
            weights = {"vector": 0.5, "text": 0.5}
            
        start_time = time.time()
        embedding = self.embedding_generator.generate_embedding(query_text)
        results = self.cosmos_db.hybrid_search(
            query_text=query_text, 
            embedding=embedding, 
            top_k=top_k, 
            min_similarity=min_similarity,
            vector_weight=weights["vector"],
            text_weight=weights["text"]
        )
        search_time = time.time() - start_time
        
        search_metrics = {
            "search_type": "Hybrid Search",
            "query_time_ms": search_time * 1000,
            "results_count": len(results),
            "embedding_dimensions": len(embedding) if embedding else 0,
            "weights": weights
        }
        
        return results, search_metrics
    
    def compare_search_methods(self, query_text, top_k=5, min_similarity=0.7):
        """Run all search methods and compare results."""
        vector_results, vector_metrics = self.vector_search(query_text, top_k, min_similarity)
        text_results, text_metrics = self.text_search(query_text, top_k)
        hybrid_results, hybrid_metrics = self.hybrid_search(query_text, top_k, min_similarity)
        
        return {
            "vector": {"results": vector_results, "metrics": vector_metrics},
            "text": {"results": text_results, "metrics": text_metrics},
            "hybrid": {"results": hybrid_results, "metrics": hybrid_metrics}
        }
    
    @staticmethod
    def calculate_result_overlap(results1, results2):
        """Calculate overlap between two result sets based on document IDs."""
        ids1 = set(doc["id"] for doc in results1)
        ids2 = set(doc["id"] for doc in results2)
        
        overlap_count = len(ids1.intersection(ids2))
        union_count = len(ids1.union(ids2))
        
        similarity = overlap_count / union_count if union_count > 0 else 0
        overlap_percentage = (overlap_count / min(len(ids1), len(ids2))) * 100 if min(len(ids1), len(ids2)) > 0 else 0
        
        return {
            "overlap_count": overlap_count,
            "jaccard_similarity": similarity,
            "overlap_percentage": overlap_percentage
        }
