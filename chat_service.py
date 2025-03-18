import time
from openai_utils import OpenAIClient
from search_service import SearchService

class ChatService:
    def __init__(self, cosmos_db_client, model_id="gpt4o_1", embedding_model="text_embedding_3_small"):
        self.model_id = model_id
        self.client = OpenAIClient(model_id=model_id)
        self.openai = self.client.get_client()
        self.deployment_name = self.client.deployment_name
        
        self.search_service = SearchService(cosmos_db_client, embedding_model=embedding_model)
        
        self.default_system_message = """You are an AI assistant helping with questions about documents stored in a database.
For each user question, relevant document excerpts will be provided as context.
Base your answers primarily on this context. If the context doesn't contain the answer, say so clearly.
Always cite your sources by mentioning the Document ID when you reference information from the context. Give the user the URL of the document if available.
Keep your answers clear, helpful, and accurate."""
    
    def format_context_for_prompt(self, results):
        """Format retrieved results as context for the prompt."""
        context_parts = []
        
        for i, result in enumerate(results):
            doc_id = result.get('sourceDocumentId', 'unknown')
            content = result.get('content', '')
            score = result.get('searchScore', 0)
            page = result.get('metadata', {}).get('page', 'unknown')
            url = result.get('metadata', {}).get('source_url', 'unknown')
            
            context_parts.append(f"[Document {i+1}] ID: '{doc_id}', Page: {page}, Relevance: {score:.2f}, URL: {url}\n{content}\n")
            
        if context_parts:
            return "Here are relevant excerpts from documents:\n\n" + "\n".join(context_parts)
        return "No relevant context found."
    
    def generate_chat_response(self, messages, context=None, stream=True):
        start_time = time.time()
        
        augmented_messages = messages.copy()
        
        if context:
            system_found = False
            for i, msg in enumerate(augmented_messages):
                if msg["role"] == "system":
                    augmented_messages[i]["content"] = f"{msg['content']}\n\n{context}"
                    system_found = True
                    break
                    
            if not system_found:
                augmented_messages.insert(0, {
                    "role": "system",
                    "content": f"{self.default_system_message}\n\n{context}"
                })
        
        token_count = 0
        response_text = ""
        
        try:
            response = self.openai.chat.completions.create(
                model=self.deployment_name,
                messages=augmented_messages,
                temperature=0.7,
                max_tokens=2000,
                stream=stream
            )
            
            if stream:
                # Process streaming response
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        token_count += 1
                        yield content, None  # Return content chunk with no error
            else:
                # Process regular response
                response_text = response.choices[0].message.content
                token_count = response.usage.completion_tokens
        
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            yield None, error_msg  # Return error
            return
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        metrics = {
            "elapsed_time_ms": elapsed_time * 1000,
            "tokens_generated": token_count,
            "model": self.model_id
        }
        
        if not stream:
            yield response_text, None  # Return full response with no error
        
        # Return final metrics after streaming is complete
        yield None, metrics
    
    def chat_with_rag(self, query, messages, search_type="hybrid", top_k=3, min_similarity=0.7, stream=True):
        # Step 1: Retrieve relevant context using the specified search method
        start_time = time.time()
        context_results = []
        search_metrics = {}
        
        if search_type == "text":
            context_results, search_metrics = self.search_service.text_search(query, top_k)
        elif search_type == "vector":
            context_results, search_metrics = self.search_service.vector_search(query, top_k, min_similarity)
        else:  # hybrid
            context_results, search_metrics = self.search_service.hybrid_search(query, top_k, min_similarity)
        
        context_retrieval_time = time.time() - start_time
        
        # Format context for the prompt
        formatted_context = self.format_context_for_prompt(context_results)
        
        # Step 2: Generate response using the context
        chat_generator = self.generate_chat_response(messages, formatted_context, stream)
        
        # Step 3: Create metrics and yield responses
        final_metrics = {
            "context_retrieval_time_ms": context_retrieval_time * 1000,
            "search_type": search_type,
            "results_retrieved": len(context_results),
            "search_metrics": search_metrics,
        }
        
        # Handle streaming or single response
        last_metrics = None
        for content, result in chat_generator:
            if content is not None:
                yield content, None  # Yield content chunk
            elif isinstance(result, dict) and "elapsed_time_ms" in result:
                # This is the final metrics result
                last_metrics = result
                final_metrics.update(result)
                yield None, final_metrics  # Yield final metrics
            elif result is not None:
                # This is an error
                yield None, result
        
        # In case we didn't get metrics from generator
        if last_metrics is None:
            yield None, final_metrics
