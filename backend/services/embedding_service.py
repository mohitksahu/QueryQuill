import chromadb
# chromadb library vector database ke liye use hoti hai

import numpy as np
# numpy numerical operations ke liye (future compatibility)

import torch
# torch GPU/CPU aur tensor operations ke liye

import gc
# gc module memory cleanup ke liye

import uuid
# uuid unique IDs generate karne ke liye

import os
# os module directories aur environment handle karta hai

from sentence_transformers import SentenceTransformer
# SentenceTransformer text ko embeddings mein convert karta hai

import config
# config file import kar rahe hain jisme model aur system settings hain


class EmbeddingService:
    # Ye class embeddings generate aur retrieve karne ka kaam karti hai

    def __init__(self):
        """Initialize the embedding service with ChromaDB 1.0.12"""

        self.device = config.DEVICE
        # CPU ya GPU device set kar raha hai

        print(f"Initializing embedding model on {self.device}")
        # Console par device info print karta hai
        
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=self.device
        )
        # SentenceTransformer model load karta hai
        
        os.makedirs(config.CHROMA_DB_DIR, exist_ok=True)
        # Ensure karta hai ki Chroma DB directory exist karti ho
        
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DB_DIR)
        )
        # Persistent ChromaDB client initialize karta hai
        
        try:
            self.collection = self.client.get_collection("documents")
            # Existing collection fetch karne ki koshish karta hai

            count = self.collection.count()
            # Documents count nikalta hai

            print(f"Connected to existing collection with {count} documents")
            # Existing collection confirmation

        except:
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            # Agar collection exist nahi karti to nayi banata hai

            print("Created new collection 'documents'")
            # New collection creation message
    

    def embed_texts(self, texts):
        """Generate embeddings for a list of texts with batch processing"""

        all_embeddings = []
        # Final embeddings store karne ke liye list

        batch_size = config.EMBEDDING_BATCH_SIZE
        # Config se batch size read karta hai
        
        for i in range(0, len(texts), batch_size):
            # Texts ko batches mein process karta hai

            batch_texts = texts[i:i+batch_size]
            # Current batch select karta hai
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True
                )
                # Batch ke embeddings generate karta hai

                all_embeddings.extend(embeddings.tolist())
                # Embeddings ko list mein add karta hai
            
            if config.CLEAR_CUDA_CACHE and torch.cuda.is_available():
                # Agar GPU available hai aur cache clear allowed hai

                torch.cuda.empty_cache()
                # CUDA cache clear karta hai

                gc.collect()
                # Python memory cleanup karta hai
        
        return all_embeddings
        # Saare embeddings return karta hai
    

    def add_documents(self, texts, metadatas):
        """Add documents to the vector database with batched processing"""

        if not texts or len(texts) == 0:
            # Agar empty input mila

            return 0
            # Kuch add nahi karega
            
        embeddings = self.embed_texts(texts)
        # Texts ke embeddings generate karta hai
        
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        # Har document ke liye unique ID generate karta hai
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        # Documents ko vector database mein add karta hai
        
        return len(texts)
        # Kitne documents add hue wo return karta hai
    

    def query(self, query_text, n_results=3):
        """Query the vector database for similar documents"""

        query_embedding = self.model.encode(
            query_text,
            convert_to_numpy=True
        ).tolist()
        # Query text ka embedding generate karta hai
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        # Vector DB mein similarity search karta hai
        
        documents = results["documents"][0]
        # Retrieved documents extract karta hai

        metadatas = results["metadatas"][0]
        # Corresponding metadata extract karta hai
        
        if config.CLEAR_CUDA_CACHE and torch.cuda.is_available():
            # GPU memory cleanup condition

            torch.cuda.empty_cache()
            # CUDA cache clear

            gc.collect()
            # Garbage collection
        
        return documents, metadatas
        # Documents aur metadata return karta hai
















'''🔍 What does this file do? (Simple Explanation)

This file defines an EmbeddingService, which is responsible for handling vector embeddings and similarity search using ChromaDB.

In simple terms, it does three main jobs:

🧠 1. Convert Text → Embeddings

Uses SentenceTransformers

Converts text documents into numerical vectors (embeddings)

Supports batch processing for performance

🗄️ 2. Store Embeddings in Vector Database

Uses ChromaDB (PersistentClient)

Stores:

Embeddings

Original text

Metadata (source, type, etc.)

Automatically creates the database if it doesn’t exist

🔎 3. Semantic Search (Similarity Query)

Converts a user query into an embedding

Finds the most similar documents using cosine similarity

Returns:

Relevant documents

Their metadata (used later in RAG)

⚡ 4. Memory Optimization

Clears CUDA cache after:

Embedding batches

Queries

Prevents GPU memory leaks

👉 This class is the retrieval engine behind your RAG system.

🧠 Simple Real-World Example

You upload:

"AI is the simulation of human intelligence"

"Machine learning is a subset of AI"

Later, user asks:

“What is artificial intelligence?”

This service:

Converts the question into an embedding

Searches the vector database

Finds the most relevant document

Sends it to the RAG pipeline'''