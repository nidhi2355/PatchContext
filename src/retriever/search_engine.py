from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

class PatchContextRetriever:
    def __init__(self):
        # 1. Use the Cloud Inference API to embed the user's question.
        # This requires ZERO local memory and avoids heavy PyTorch downloads.
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction", # Required task definition for embeddings
            huggingfacehub_api_token=os.environ.get("HF_TOKEN") 
        )
        
        # 2. Load the offline FAISS index we built locally
        self.vectorstore = FAISS.load_local(
            "faiss_index", 
            self.embeddings, 
            allow_dangerous_deserialization=True 
        )
        
        # 3. Standard retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4}
        )

    def retrieve_relevant_chunks(self, query):
        print(f"Executing lightweight search for query: '{query}'")
        docs = self.retriever.invoke(query)
        return docs

if __name__ == "__main__":
    # Test our search pipeline with a standard repo exploration question
    engine = PatchContextRetriever()
    test_query = "Why was dependency injection redesigned?"
    
    results = engine.retrieve_relevant_chunks(test_query)
    
    print("\n--- Retrieval Engine Test Results ---")
    for i, doc in enumerate(results):
        print(f"\n[Rank {i+1}] Type: {doc.metadata.get('type').upper()}")
        print(f"Source Link: {doc.metadata.get('source')}")
        print(f"Snippet Preview:\n{doc.page_content[:250]}...")