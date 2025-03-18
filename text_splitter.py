import tiktoken

class TextSplitter:
    def __init__(self, tokens_per_chunk=500, overlap_tokens=50):
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def split_pages(self, pages):
        """Split pages directly from pymupdf4llm output"""
        chunks = []
        
        for page in pages:
            page_num = page['metadata']['page']
            content = page['text']
            tokens = self.encoding.encode(content)
            
            start = 0
            while start < len(tokens):
                end = start + self.tokens_per_chunk
                chunk_tokens = tokens[start:min(end, len(tokens))]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "page": page_num
                    }
                })
                start = start + self.tokens_per_chunk - self.overlap_tokens
                
        print(f"Split {len(pages)} pages into {len(chunks)} chunks with overlap of {self.overlap_tokens} tokens.")
        return chunks
