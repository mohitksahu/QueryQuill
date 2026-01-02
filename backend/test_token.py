# Ye file Hugging Face token ko test karne ke liye hai

import os
# os module environment variables access karne ke kaam aata hai
import sys
# sys module system-related operations ke liye hota hai (yahan directly use nahi ho raha)
from pathlib import Path
# Path class file paths ko safely handle karne ke liye use hoti hai
from dotenv import load_dotenv
# load_dotenv function .env file se environment variables load karta hai
from huggingface_hub import HfApi
# HfApi Hugging Face ke API ke saath interact karne ke liye use hota hai

def test_token():
    # Ye function Hugging Face token ko test karta hai
    env_path = Path(r".env")
    # .env file ka path define kiya gaya hai (same folder mein)
    if not env_path.exists():
        # Check karta hai ki .env file exist karti hai ya nahi
        print(f"❌ Error: .env file not found at {env_path}")
        # Agar file nahi mili to error message print karega
        return False
        # Program yahin stop ho jayega

    print(f"Loading .env from: {env_path}")
    # Batata hai ki .env file load ho rahi hai
    load_dotenv(dotenv_path=env_path)
    # .env file ke andar ke variables environment mein load karta hai
    token = os.getenv("HUGGINGFACE_TOKEN")
    # Environment se HUGGINGFACE_TOKEN ko read karta hai

    if not token:
        # Agar token nahi mila

        print("❌ Error: HUGGINGFACE_TOKEN not found in .env file")
        # Error message print karega

        return False
        # Function yahin return ho jayega

    try:
        # Error handling start hoti hai

        api = HfApi(token=token)
        # Hugging Face API object create karta hai token ke saath

        user_info = api.whoami()
        # Token ke owner ki information nikalta hai

        print(f"✓ Successfully authenticated as: {user_info['name']}")
        # Username print karta hai agar authentication successful ho

        print("Checking access to Llama 2...")
        # Batata hai ki Llama 2 model access check ho raha hai

        model_info = api.model_info("meta-llama/Llama-2-7b-chat-hf")
        # Llama-2-7B-Chat model ki information fetch karta hai

        print(f"✓ You have access to: {model_info.modelId}")
        # Confirm karta hai ki model accessible hai

        print("✓ Your token is valid and has proper permissions!")
        # Final success message

        return True
        # Function successful completion ke baad True return karta hai

    except Exception as e:
        # Agar koi error aata hai

        print(f"❌ Error: {e}")
        # Error message print karta hai

        return False
        # Failure ke case mein False return karta hai


if __name__ == "__main__":
    # Ye check karta hai ki file directly run ho rahi hai ya import hui hai

    test_token()
    # test_token function ko call karta hai
