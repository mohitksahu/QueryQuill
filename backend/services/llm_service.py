import torch
# torch library GPU/CPU aur tensor operations ke liye use hoti hai

import gc
# gc module unused memory clean karne ke liye hota hai

import config
# config file import kar rahe hain jisme device aur model settings hain

import os
# os module environment variables read karne ke liye use hota hai

import sys
# sys module runtime environment detect karne ke liye use hota hai

from huggingface_hub import login
# Hugging Face ke saath authentication ke liye login function

from utils.env_utils import load_environment
# .env file se environment variables load karne ke liye


# Dynamic imports based on availability
try:
    from transformers import BitsAndBytesConfig
    # BitsAndBytesConfig quantization ke liye use hota hai

    QUANTIZATION_AVAILABLE = True
    # Flag batata hai ki quantization available hai

except ImportError:
    QUANTIZATION_AVAILABLE = False
    # Agar bitsandbytes install nahi hai

    print("Warning: BitsAndBytesConfig not available, quantization disabled")
    # Warning print karta hai


from transformers import AutoTokenizer, AutoModelForCausalLM
# Tokenizer aur causal language model load karne ke liye


class LLMService:
    # Ye class poora LLM lifecycle handle karti hai

    def __init__(self):
        """Initialize the LLM service with automatic device detection"""

        print(f"Initializing LLM on {config.DEVICE}")
        # CPU ya GPU par model load ho raha hai

        print(f"Model: {config.MODEL_NAME}")
        # Kaunsa model load ho raha hai

        print(f"Quantization enabled: {config.USE_QUANTIZATION}")
        # Quantization enabled hai ya nahi

        load_environment()
        # .env file load karta hai

        token = os.getenv("HUGGINGFACE_TOKEN")
        # Hugging Face token environment se read karta hai

        if token:
            print("Using token from .env file")
            # Token mil gaya

            try:
                login(token=token)
                # Hugging Face login attempt

                print("Login successful")
                # Login success

            except Exception as e:
                print(f"Login failed: {e}")
                # Login error print karta hai
        else:
            print("Warning: No Hugging Face token found in .env file")
            # Token missing warning

        self._load_model(token)
        # Tokenizer aur model load karne ke liye internal method call


    def _load_model(self, token):
        """Load the model with appropriate configuration for the device"""

        try:
            print("Loading tokenizer...")
            # Tokenizer load hone wala hai

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_NAME,
                token=token if token else None,
                trust_remote_code=True
            )
            # Hugging Face se tokenizer load karta hai

            if self.tokenizer.pad_token is None:
                # Agar pad token missing hai

                self.tokenizer.pad_token = self.tokenizer.eos_token
                # EOS token ko pad token bana deta hai

                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                # Pad token ID set karta hai

            model_kwargs = {
                "pretrained_model_name_or_path": config.MODEL_NAME,
                # Model ka naam

                "torch_dtype": torch.float16 if config.USE_GPU else torch.float32,
                # GPU ke liye fp16, CPU ke liye fp32

                "trust_remote_code": True,
                # Custom model code allow karta hai

                "low_cpu_mem_usage": True
                # CPU memory optimization
            }

            if token:
                model_kwargs["token"] = token
                # Token add karta hai agar available ho

            if config.USE_QUANTIZATION and QUANTIZATION_AVAILABLE:
                print("Configuring 4-bit quantization...")
                # Quantization setup start

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    # 4-bit quantization enable

                    bnb_4bit_use_double_quant=True,
                    # Double quantization enable

                    bnb_4bit_quant_type="nf4",
                    # NF4 quantization type

                    bnb_4bit_compute_dtype=torch.float16 if config.USE_GPU else torch.float32
                    # Compute dtype
                )

                model_kwargs["quantization_config"] = quantization_config
                # Quantization config attach karta hai

                if config.USE_GPU and torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"
                    # Automatic GPU mapping

                    if not config.IN_COLAB:
                        model_kwargs["max_memory"] = {0: "5GiB"}
                        # Local GPU memory limit

                else:
                    model_kwargs["device_map"] = {"": "cpu"}
                    # CPU mapping

            else:
                print("Quantization disabled or not available - loading model in full precision")

                if config.USE_GPU and torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"

                    if not config.IN_COLAB:
                        model_kwargs["max_memory"] = {0: "5GiB"}
                else:
                    model_kwargs["device_map"] = {"": "cpu"}

            print("Loading model (this may take a few minutes)...")
            # Actual model loading start

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            # Model load karta hai

            if not config.USE_GPU or not torch.cuda.is_available():
                self.device = "cpu"
                self.model = self.model.to("cpu")
                # CPU par model move karta hai
            else:
                self.device = "cuda"
                # GPU use ho raha hai

            print(f"✓ Model loaded successfully on {self.device}!")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            # Error print karta hai

            raise RuntimeError(
                "Failed to load the Llama model. Please check your environment and dependencies."
            )
            # Hard failure raise karta hai


    def generate_response(self, prompt, max_new_tokens=None):
        """Generate a response from the LLM given a prompt"""

        if max_new_tokens is None:
            max_new_tokens = config.MAX_NEW_TOKENS
            # Default token limit use karta hai

        if hasattr(config, 'MAX_INPUT_LENGTH'):
            prompt = prompt[-config.MAX_INPUT_LENGTH:]
            # Prompt ko max length tak trim karta hai

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            # Prompt ko tokens mein convert karta hai

            if hasattr(self, 'device'):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # Inputs ko correct device par move karta hai

            elif config.USE_GPU and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
                # Text generate karta hai

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            # Sirf new generated text decode karta hai

            if (
                hasattr(config, 'CLEAR_CUDA_CACHE')
                and config.CLEAR_CUDA_CACHE
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
                gc.collect()
                # GPU memory clear karta hai

            return response.strip()
            # Final clean response return karta hai

        except Exception as e:
            print(f"Error generating response: {e}")
            # Error print karta hai

            return f"Error generating response: {str(e)}"
            # Error message return karta hai


    def get_model_info(self):
        """Get information about the loaded model"""

        return {
            "model_name": config.MODEL_NAME,
            "device": getattr(self, 'device', config.DEVICE),
            "use_gpu": config.USE_GPU,
            "use_quantization": config.USE_QUANTIZATION,
            "in_colab": config.IN_COLAB,
            "cuda_available": torch.cuda.is_available(),
            "quantization_available": QUANTIZATION_AVAILABLE
        }
        # Model diagnostics return karta hai
