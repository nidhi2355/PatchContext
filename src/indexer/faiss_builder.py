import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Import the processor we just built in Step 3
from src.indexer.data_processor import RepositoryDataProcessor

class VectorStoreBuilder:
    def __init__(self, index_save_path: str = "data/faiss_index"):
        self.index_save_path = index_save_path
        
        print("Initializing BAAI/bge-small-en-v1.5 embedding model...")
        # Using the exact free, high-performance model specified in the architecture
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have a compatible NVIDIA GPU
            encode_kwargs={'normalize_embeddings': True}
        )

    def build_and_save_index(self):
        """Processes raw repository chunks and builds a FAISS vector database."""
        # 1. Initialize our processor and get the semantic chunks
        processor = RepositoryDataProcessor()
        chunks = processor.process_and_chunk_repository()
        
        if not chunks:
            print("No chunks available to embed. Exiting.")
            return

        print(f"Embedding {len(chunks)} chunks into FAISS vector space...")
        # 2. Build the FAISS index from the documents
        # This will download the BGE model weights on the first run, which takes a moment.
        # It then processes the text locally to create the vectors.
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        
        # 3. Save the index to disk for persistent storage
        print(f"Saving FAISS index to {self.index_save_path}...")
        vectorstore.save_local(self.index_save_path)
        print("Vector database successfully built and saved!")

if __name__ == "__main__":
    # Ensure the destination folder exists
    os.makedirs("data/faiss_index", exist_ok=True)
    
    builder = VectorStoreBuilder()
    builder.build_and_save_index()