import os
from dotenv import load_dotenv
from openai_utils import OpenAIClient

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, model_id="text_embedding_3_large"):
        """Initialize the EmbeddingGenerator with the specified model ID."""
        self.client_data = OpenAIClient(model_id)
        self.client = self.client_data.get_client()
        self.deployment_name = self.client_data.deployment_name

    def generate_embedding(self, text):
        """Generate an embedding for a single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.deployment_name
        )
        print(f"Generated embedding for text of length {len(text)}.")
        return response.data[0].embedding

    def generate_embeddings_for_chunks(self, chunks):
        """Generate embeddings for a list of text chunks."""
        embeddings = [self.generate_embedding(chunk) for chunk in chunks]
        print(f"Generated embeddings for {len(chunks)} chunks.")
        return embeddings
