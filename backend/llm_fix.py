"""Fix for the 'dict' object has no attribute 'input_ids' error in LLM service."""
# Ye file LLM service mein aane wale input_ids error ko fix karne ke liye hai

import sys
# sys module system-level operations ke liye hota hai (yahan future compatibility ke liye)

from pathlib import Path
# Path class file paths ko safely handle karne ke liye use hoti hai


def main():
    """Fix the LLM service error."""
    # Ye main function LLM service file ko fix karega

    llm_service_path = Path(__file__).parent / "services" / "llm_service.py"
    # llm_service.py ka exact path define kiya gaya hai

    if not llm_service_path.exists():
        # Check karta hai ki llm_service.py file exist karti hai ya nahi

        print(f"Error: LLM service file not found at {llm_service_path}")
        # Agar file nahi mili to error message print karta hai

        return
        # Program yahin stop ho jata hai

    content = llm_service_path.read_text()
    # LLM service file ka poora content read karta hai

    if "inputs.input_ids" in content or "inputs.attention_mask" in content:
        # Check karta hai ki galat attribute access use ho raha hai ya nahi

        content = content.replace("inputs.input_ids", "inputs['input_ids']")
        # inputs.input_ids ko sahi dictionary access se replace karta hai

        content = content.replace("inputs.attention_mask", "inputs['attention_mask']")
        # inputs.attention_mask ko bhi sahi format mein replace karta hai

        llm_service_path.write_text(content)
        # Fixed content ko file mein wapas write karta hai

        print(f"✅ Fixed LLM service file: {llm_service_path}")
        # Success message print karta hai

    else:
        # Agar koi issue mila hi nahi

        print(f"⚠️ No issues found in LLM service file")
        # Warning message print karta hai


if __name__ == "__main__":
    # Check karta hai ki file directly run ho rahi hai

    main()
    # main function ko call karta hai
