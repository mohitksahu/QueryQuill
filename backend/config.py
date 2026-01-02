import os
# os module environment variables access karne ke kaam aata hai

from pathlib import Path
# Path class file paths ko safely handle karne ke liye use hoti hai


# Find the project root directory
BASE_DIR = Path(__file__).resolve().parent
# Current file ka directory path nikal raha hai

PROJECT_ROOT = BASE_DIR.parent
# Project ka root folder define kar raha hai


# Flask settings
HOST = os.environ.get('HOST', '0.0.0.0')
# Flask server kis IP par chalega (default: sab interfaces)

PORT = int(os.environ.get('PORT', 5000))
# Flask server ka port (default: 5000)

DEBUG = os.environ.get('DEBUG', 'True') == 'True'
# Debug mode environment variable se read ho raha hai


# Path settings
DATA_DIR = PROJECT_ROOT / 'data'
# Data folder ka path define kar raha hai

UPLOAD_FOLDER = DATA_DIR / 'uploaded'
# Uploaded files ke liye folder path

CHROMA_DB_DIR = DATA_DIR / 'chroma_db'
# Chroma vector database ka folder path


# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
# Agar upload folder nahi hai to create kar dega

CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
# Agar chroma_db folder nahi hai to create kar dega


# Model settings - auto-detect and support both CPU and GPU
import sys
# sys module runtime environment detect karne ke liye use hota hai

import torch
# torch library GPU/CPU availability check karne ke kaam aati hai


# Check if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules
# Agar Colab environment hai to True ho jayega


# Auto-detect device capabilities
def get_device_config():
    # Ye function device configuration decide karta hai

    if torch.cuda.is_available() and not IN_COLAB:
        # Agar local GPU available hai aur Colab nahi hai

        return {
            "device": "cuda",
            # GPU use karega

            "use_quantization": True,
            # GPU ke saath quantization enable karega

            "model_name": "meta-llama/Llama-2-7b-chat-hf"
            # LLM model ka naam
        }

    else:
        # CPU environment ya Colab ke case mein

        return {
            "device": "cpu",
            # CPU use karega

            "use_quantization": True,
            # CPU ke liye bhi quantization enable karega

            "model_name": "meta-llama/Llama-2-7b-chat-hf"
            # Same LLM model use karega
        }


# Get device configuration
device_config = get_device_config()
# Device config function ko call kar raha hai


MODEL_NAME = device_config["model_name"]
# Model ka naam set kar raha hai

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Embedding model ka naam define kar raha hai

USE_GPU = device_config["device"] == "cuda"
# Check kar raha hai GPU use ho raha hai ya nahi

DEVICE = device_config["device"]
# Final device (cpu / cuda)

USE_QUANTIZATION = device_config["use_quantization"]
# Quantization enable hai ya nahi


print(f"🔧 Device Configuration:")
# Device configuration heading print karta hai

print(f"   - Device: {DEVICE}")
# CPU ya GPU print karta hai

print(f"   - Model: {MODEL_NAME}")
# LLM model ka naam print karta hai

print(f"   - Quantization: {USE_QUANTIZATION}")
# Quantization status print karta hai

print(f"   - In Colab: {IN_COLAB}")
# Colab environment status print karta hai


# Memory optimization settings
QUANTIZATION = "4bit"
# Quantization type define karta hai

MAX_NEW_TOKENS = 256
# LLM kitne naye tokens generate karega

TEMPERATURE = 0.7
# Response randomness control karta hai

TOP_P = 0.9
# Nucleus sampling ka parameter

MAX_INPUT_LENGTH = 512
# Input prompt ka maximum token length

EMBEDDING_BATCH_SIZE = 8
# Embeddings batch size define karta hai

CHUNK_SIZE = 600
# RAG ke liye document chunk size

CHUNK_OVERLAP = 50
# Chunks ke beech overlap define karta hai

CLEAR_CUDA_CACHE = True
# GPU memory clear karne ka option
