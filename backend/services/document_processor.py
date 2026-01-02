import os
# os module file paths aur file size check karne ke liye

import PyPDF2
# PyPDF2 PDF files se text extract karne ke liye use hota hai

from PIL import Image
# PIL image files open karne ke liye

import pytesseract
# pytesseract images se OCR ke through text nikalta hai

from langchain.text_splitter import RecursiveCharacterTextSplitter
# Text ko chunks mein todne ke liye LangChain ka splitter

from services.embedding_service import EmbeddingService
# EmbeddingService jo text ko vector DB mein store karta hai

import torch
# torch GPU availability aur CUDA memory handle karta hai

import gc
# gc unused memory clean karta hai

import config
# config file jisme chunk size aur memory settings defined hain


class DocumentProcessor:
    # Ye class documents ko process karke embeddings mein convert karti hai

    def __init__(self):
        """Initialize the document processor"""

        self.embedding_service = EmbeddingService()
        # Embedding service ka object create karta hai

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            # Har chunk ka size define karta hai

            chunk_overlap=config.CHUNK_OVERLAP,
            # Chunks ke beech overlap define karta hai

            separators=["\n\n", "\n", ".", " ", ""]
            # Text todne ke priority separators
        )
    

    def process(self, file_path, document_type):
        """Process a document based on its type"""

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        # File ka size MB mein calculate karta hai
        
        if file_size_mb > 50:
            # Agar file 50MB se badi hai

            raise ValueError(
                f"File too large ({file_size_mb:.1f}MB). Maximum size is 50MB."
            )
            # Error throw karta hai
            
        if document_type == 'pdf':
            # Agar document PDF hai

            return self.process_pdf(file_path)
            # PDF processing function call karta hai

        elif document_type in ['jpg', 'jpeg', 'png', 'gif']:
            # Agar document image hai

            return self.process_image(file_path)
            # Image processing function call karta hai

        else:
            # Unsupported document type

            raise ValueError(f"Unsupported document type: {document_type}")
    

    def process_pdf(self, file_path):
        """Process PDF with memory management"""

        processed_chunks = 0
        # Total processed chunks count

        file_name = os.path.basename(file_path)
        # File ka naam nikalta hai
        
        with open(file_path, 'rb') as file:
            # PDF file open karta hai

            pdf_reader = PyPDF2.PdfReader(file)
            # PDF reader object banata hai
            
            batch_size = 5
            # Ek baar mein kitne pages process honge

            total_pages = len(pdf_reader.pages)
            # Total pages count
            
            for batch_start in range(0, total_pages, batch_size):
                # Pages ko batches mein loop karta hai

                batch_end = min(batch_start + batch_size, total_pages)
                # Batch ka end page

                batch_text = ""
                # Batch ka text yahan store hoga
                
                for i in range(batch_start, batch_end):
                    # Har page ke liye loop

                    text = pdf_reader.pages[i].extract_text()
                    # Page se text extract karta hai

                    if text:
                        batch_text += f"Page {i+1}:\n{text}\n\n"
                        # Page number ke saath text add karta hai
                
                if batch_text:
                    # Agar batch mein text mila

                    chunks = self.text_splitter.split_text(batch_text)
                    # Text ko chunks mein todta hai
                    
                    metadatas = [{
                        "source": file_name,
                        "type": "pdf",
                        "page_range": f"{batch_start+1}-{batch_end}"
                    } for _ in chunks]
                    # Har chunk ke liye metadata banata hai
                    
                    processed_chunks += self.embedding_service.add_documents(
                        chunks,
                        metadatas
                    )
                    # Chunks ko vector DB mein add karta hai
                    
                    if config.CLEAR_CUDA_CACHE and torch.cuda.is_available():
                        # Agar GPU memory cleanup enabled hai

                        torch.cuda.empty_cache()
                        # CUDA cache clear karta hai

                        gc.collect()
                        # Python memory clean karta hai
        
        return processed_chunks
        # Total processed chunks return karta hai
    

    def process_image(self, file_path):
        """Extract text from image using OCR and add to vector database"""

        file_name = os.path.basename(file_path)
        # Image file ka naam
        
        try:
            image = Image.open(file_path)
            # Image open karta hai

            text = pytesseract.image_to_string(image)
            # OCR ke through text extract karta hai
            
            if text.strip():
                # Agar image se text mila

                chunks = self.text_splitter.split_text(text)
                # Text ko chunks mein todta hai
                
                metadatas = [{
                    "source": file_name,
                    "type": "image"
                } for _ in chunks]
                # Metadata banata hai
                
                return self.embedding_service.add_documents(
                    chunks,
                    metadatas
                )
                # Chunks ko vector DB mein add karta hai
            
            return 0
            # Agar koi text nahi mila

        except Exception as e:
            # Agar OCR ya image processing fail ho jaye

            raise ValueError(f"Error processing image: {str(e)}")
            # Clear error message throw karta hai




'''🔍 What does this file do? (Simple Explanation)

This file defines a DocumentProcessor, whose job is to:

👉 Take uploaded documents (PDFs or Images)
👉 Extract text from them
👉 Split the text into chunks
👉 Convert those chunks into embeddings
👉 Store them in the vector database (ChromaDB)

This is the ingestion pipeline of your RAG system.

🧠 What types of documents does it support?
📄 PDF Files

Uses PyPDF2 to extract text page by page

Processes PDFs in small batches to save memory

Adds page number metadata

🖼️ Image Files (JPG, PNG, etc.)

Uses OCR (pytesseract) to extract text from images

Useful for scanned documents

⚙️ Key Features

✅ File size limit (50MB)
✅ Memory-efficient batch processing
✅ Text chunking for RAG
✅ Metadata for citation
✅ CUDA memory cleanup

🧠 Simple Real-World Example

You upload:

ml_notes.pdf (20 pages)

This class will:

Read 5 pages at a time

Extract text

Split text into chunks (600 chars each)

Store embeddings with metadata like:

{
  "source": "ml_notes.pdf",
  "type": "pdf",
  "page_range": "1-5"
}


Later, RAG uses these chunks to answer questions.'''