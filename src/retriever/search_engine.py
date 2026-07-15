import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

class PatchContextRetriever:
    def __init__(self, index_path: str = "data/faiss_index"):
        self.index_path = index_path
        
        # Initialize the same embedding model used during indexing
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load the local FAISS database
        print("Loading local FAISS vector index...")
        # allow_dangerous_deserialization is required to load local pickle files safely
        self.vectorstore = FAISS.load_local(
            self.index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Load the Cross-Encoder model for high-precision reranking
        print("Loading Cross-Encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    def retrieve_relevant_chunks(self, query: str, top_k_mmr: int = 10, top_k_rerank: int = 4):
        """Retrieves diverse chunks using MMR and filters them with a Cross-Encoder."""
        
        # 1. Configure the vectorstore as an MMR retriever
        # fetch_k defines how many total items to pull initially before filtering down to top_k_mmr diverse ones
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k_mmr, "fetch_k": 20}
        )
        
        print(f"Executing MMR search for query: '{query}'")
        initial_docs = retriever.invoke(query)
        
        if not initial_docs:
            print("No matching documents found in vector store.")
            return []
            
        # 2. Prepare inputs for Cross-Encoder Reranking
        # The cross-encoder takes a list of pairs: [[query, document_1], [query, document_2], ...]
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        print("Reranking chunks via Cross-Encoder...")
        scores = self.reranker.predict(pairs)
        
        # Pair up the documents with their corresponding cross-encoder score
        scored_docs = list(zip(initial_docs, scores))
        
        # Sort documents by score in descending order (highest relevance first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the final top-k reranked documents
        final_docs = [doc for doc, score in scored_docs[:top_k_rerank]]
        print(f"Successfully retrieved and reranked top {len(final_docs)} context chunks.")
        
        return final_docs

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