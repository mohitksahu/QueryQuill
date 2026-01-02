from services.embedding_service import EmbeddingService
# EmbeddingService import kar rahe hain jo vector DB se documents retrieve karta hai

import torch
# torch library GPU availability aur CUDA memory handle karne ke liye use hoti hai

import gc
# gc module memory cleanup (garbage collection) ke kaam aata hai

import config
# config file import kar rahe hain jisme system settings defined hain


class RAGPipeline:
    # Ye class poora RAG (Retrieval-Augmented Generation) flow handle karti hai

    def __init__(self, llm_service):
        """Initialize the RAG pipeline"""
        # Constructor jo RAG pipeline ko initialize karta hai

        self.llm_service = llm_service
        # LLM service ka reference store kar raha hai

        self.embedding_service = EmbeddingService()
        # Embedding service ka object create kar raha hai
    

    def process_query(self, query):
        """Process a query through the RAG pipeline"""
        # Ye function user ke question ko process karta hai

        documents, metadatas = self.embedding_service.query(query, n_results=3)
        # Vector database se top 3 relevant documents retrieve karta hai
        
        context_parts = []
        # Context ke alag-alag parts store karne ke liye list
        
        for i, doc in enumerate(documents):
            # Har retrieved document ke liye loop

            if len(doc) > 300:
                # Agar document bahut lamba hai

                doc = doc[:300] + "..."
                # To usko truncate karke short bana deta hai

            context_parts.append(f"Document {i+1}: {doc}")
            # Document ko numbered format mein add karta hai
        
        context = "\n\n".join(context_parts)
        # Saare documents ko ek single context string bana deta hai
        
        prompt = f"""You are a helpful educational assistant. Use only the following context to answer the student's question. If you don't know the answer based on the context, say that you don't have enough information.

Context:
{context}

Student Question: {query}

Helpful Answer:"""
        # LLM ke liye final prompt prepare karta hai (context + question)
        
        response = self.llm_service.generate_response(prompt)
        # LLM ko prompt bhej kar answer generate karta hai
        
        sources = [
            {
                "source": metadata["source"],
                "type": metadata["type"]
            } for metadata in metadatas
        ]
        # Sources ko clean format mein prepare karta hai
        
        if config.CLEAR_CUDA_CACHE and torch.cuda.is_available():
            # Agar config allow karta hai aur GPU available hai

            torch.cuda.empty_cache()
            # CUDA cache clear karta hai

            gc.collect()
            # Python memory cleanup karta hai
        
        return response, sources
        # Final answer aur uske sources return karta hai
