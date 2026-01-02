from services.llm_service import LLMService
# LLMService ko import kar rahe hain jo language model handle karta hai

from services.embedding_service import EmbeddingService
# EmbeddingService ko import kar rahe hain jo text ko embeddings mein convert karta hai

from services.rag_pipeline import RAGPipeline
# RAGPipeline ko import kar rahe hain jo retrieval + generation ka kaam karta hai

from utils.env_utils import load_environment
# load_environment function environment variables load karne ke liye use hota hai


def test_backend():
    # Ye function poore backend ka test run karega

    print("Testing backend components...")
    # Console par batata hai ki backend testing start ho rahi hai
    
    load_environment()
    # .env ya system se environment variables load karta hai
    
    print("\n--- Testing LLM Service ---")
    # LLM service testing ka section start ho raha hai
    
    llm = LLMService()
    # LLMService ka object create kiya ja raha hai
    
    test_prompt = "Explain what RAG means in AI"
    # Ek test prompt define kiya gaya hai
    
    print(f"Sending test prompt: '{test_prompt}'")
    # Prompt console par show karta hai
    
    response = llm.generate_response(test_prompt)
    # LLM ko prompt bhej kar response generate karta hai
    
    print(f"Response: {response[:100]}...")
    # Response ke sirf pehle 100 characters print karta hai
    

    print("\n--- Testing Embedding Service ---")
    # Embedding service testing ka section start
    
    embed_service = EmbeddingService()
    # EmbeddingService ka object create karta hai
    
    test_texts = ["This is a test document about artificial intelligence."]
    # Test ke liye ek sample document define karta hai
    
    test_metadata = [{"source": "test.txt", "type": "text"}]
    # Document ke saath metadata attach karta hai
    
    print("Adding test document to vector database...")
    # Batata hai ki document vector DB mein add ho raha hai
    
    embed_service.add_documents(test_texts, test_metadata)
    # Text ko embeddings mein convert karke store karta hai
    

    print("\n--- Testing RAG Pipeline ---")
    # RAG pipeline testing ka section start
    
    rag = RAGPipeline(llm)
    # RAGPipeline ko LLM ke saath initialize karta hai
    
    test_query = "What is artificial intelligence?"
    # Ek test query define karta hai
    
    print(f"Sending query to RAG pipeline: '{test_query}'")
    # Query console par print karta hai
    
    result, sources = rag.process_query(test_query)
    # Query ko RAG pipeline mein bhejta hai aur response + sources leta hai
    
    print(f"Response: {result[:100]}...")
    # RAG response ke pehle 100 characters print karta hai
    
    print(f"Sources: {sources}")
    # Batata hai ki answer kis source se aaya
    

    print("\nBackend test completed!")
    # Backend testing successfully complete hone ka message


if __name__ == "__main__":
    # Check karta hai ki file directly run ho rahi hai
    
    test_backend()
    # test_backend function ko call karta hai
