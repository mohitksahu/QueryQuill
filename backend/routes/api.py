from flask import Blueprint, request, jsonify
# Flask ke Blueprint aur request/response utilities

from services.llm_service import LLMService
# LLMService jo language model handle karta hai

from services.document_processor import DocumentProcessor
# DocumentProcessor jo files ko process karta hai

from services.rag_pipeline import RAGPipeline
# RAGPipeline jo retrieval + generation ka flow handle karta hai

import os
# os module file handling ke liye

import config
# config file jisme model aur path settings defined hain

import time
# time module response time measure karne ke liye

import torch
# torch GPU availability aur memory handle karta hai


# Initialize services
api_bp = Blueprint('api', __name__)
# Flask Blueprint create kar rahe hain API routes ke liye


# Lazy initialization to save memory
llm_service = None
document_processor = None
rag_pipeline = None
# Initially services None rakhe gaye hain


def get_llm_service():
    # Ye function LLMService ko lazy-load karta hai

    global llm_service
    # Global variable access

    if llm_service is None:
        # Agar LLM abhi create nahi hua

        llm_service = LLMService()
        # To naya LLMService object banata hai

    return llm_service
    # LLMService return karta hai


def get_document_processor():
    # DocumentProcessor ke liye lazy loader

    global document_processor

    if document_processor is None:
        document_processor = DocumentProcessor()
        # Pehli baar DocumentProcessor create hota hai

    return document_processor


def get_rag_pipeline():
    # RAGPipeline ke liye lazy loader

    global rag_pipeline

    if rag_pipeline is None:
        rag_pipeline = RAGPipeline(get_llm_service())
        # RAGPipeline ko LLM ke saath initialize karta hai

    return rag_pipeline


@api_bp.route('/query', methods=['POST'])
def query():
    """API endpoint for querying the RAG model"""

    start_time = time.time()
    # Request start hone ka time record karta hai

    data = request.json
    # JSON body read karta hai

    query = data.get('query', '')
    # Query text extract karta hai
    
    if not query:
        # Agar query empty hai

        return jsonify({"error": "No query provided"}), 400
        # Bad request return karta hai
    
    try:
        if torch.cuda.is_available():
            # Agar GPU available hai

            torch.cuda.empty_cache()
            # Purana GPU memory clear karta hai
        
        pipeline = get_rag_pipeline()
        # RAG pipeline fetch karta hai

        response, sources = pipeline.process_query(query)
        # Query ko RAG pipeline se process karta hai

        processing_time = time.time() - start_time
        # Total processing time calculate karta hai
        
        return jsonify({
            "response": response,
            "sources": sources,
            "processing_time": round(processing_time, 2)
        })
        # Final response JSON return karta hai

    except Exception as e:
        # Agar koi error aaye

        return jsonify({"error": str(e)}), 500
        # Internal server error return karta hai


@api_bp.route('/upload', methods=['POST'])
def upload_document():
    """API endpoint for uploading and processing documents"""

    if 'file' not in request.files:
        # Agar file hi nahi bheji

        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    # Uploaded file object
    
    if file.filename == '':
        # Agar filename empty hai

        return jsonify({"error": "No selected file"}), 400
    
    try:
        if torch.cuda.is_available():
            # GPU cache clear before processing

            torch.cuda.empty_cache()
            
        file_path = os.path.join(
            str(config.UPLOAD_FOLDER),
            file.filename
        )
        # Upload folder ke andar file ka path

        file.save(file_path)
        # File disk par save karta hai
        
        document_type = file.filename.split('.')[-1].lower()
        # File extension se document type nikalta hai

        processor = get_document_processor()
        # DocumentProcessor fetch karta hai

        processed_chunks = processor.process(
            file_path,
            document_type
        )
        # File ko process karta hai (PDF/Image)
        
        return jsonify({
            "success": True,
            "message": f"File {file.filename} uploaded and processed successfully",
            "chunks_processed": processed_chunks
        })
        # Success response return karta hai

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        # Error response


@api_bp.route('/status', methods=['GET'])
def status():
    """API endpoint for checking server status"""

    gpu_available = torch.cuda.is_available()
    # GPU available hai ya nahi

    gpu_info = {}
    # GPU details ke liye empty dict
    
    if gpu_available:
        # Agar GPU available hai

        gpu_info = {
            "device_count": torch.cuda.device_count(),
            # Total GPU count

            "current_device": torch.cuda.current_device(),
            # Current active GPU

            "device_name": torch.cuda.get_device_name(0),
            # GPU ka naam

            "memory_allocated_mb": round(
                torch.cuda.memory_allocated() / (1024 ** 2), 2
            ),
            # GPU memory allocated

            "memory_reserved_mb": round(
                torch.cuda.memory_reserved() / (1024 ** 2), 2
            )
            # GPU memory reserved
        }
    
    return jsonify({
        "status": "online",
        # Server status

        "gpu_available": gpu_available,
        # GPU availability

        "gpu_info": gpu_info,
        # GPU details

        "model_name": config.MODEL_NAME,
        # Loaded LLM model

        "embedding_model": config.EMBEDDING_MODEL
        # Embedding model name
    })


'''🔍 What does this file do? (Simple Explanation)

This file defines the Flask API layer for your AI + RAG backend.

It exposes three REST API endpoints that your frontend (or Postman) can call:

🔹 1️⃣ /query – Ask a Question (RAG)

Takes a user query

Uses:

Embeddings

Vector database

LLM (via RAGPipeline)

Returns:

AI-generated answer

Sources used

Processing time

👉 This is your main chat / Q&A endpoint

🔹 2️⃣ /upload – Upload Documents

Accepts PDFs or images

Saves them to disk

Extracts text

Converts text → embeddings

Stores them in ChromaDB

👉 This is your knowledge ingestion endpoint

🔹 3️⃣ /status – Health Check

Tells if the server is running

Shows:

GPU availability

GPU memory usage

Loaded models

👉 Useful for monitoring & debugging

⚙️ Key Design Decisions
✅ Lazy Initialization

LLM, RAG, and DocumentProcessor are created only when needed

Saves RAM and startup time

✅ GPU Memory Safety

Clears CUDA cache before heavy operations

Prevents out-of-memory crashes

✅ Modular Services

LLM logic → LLMService

Ingestion → DocumentProcessor

RAG logic → RAGPipeline

🧠 Simple Real-World Example
Upload a document
POST /upload


Response:

{
  "success": true,
  "chunks_processed": 42
}

Ask a question
POST /query
{
  "query": "Explain artificial intelligence"
}


Response:

{
  "response": "...",
  "sources": [...],
  "processing_time": 1.84
}'''