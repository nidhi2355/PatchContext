import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Allowed extensions for codebase and documentation indexing
ALLOWED_EXTENSIONS = {".py", ".md", ".txt", ".html", ".js", ".json"}

# Directories to skip during ingestion
IGNORE_DIRS = {
    ".git", "venv", "env", "__pycache__", "node_modules", "faiss_index",
    # Non-English doc translation folders to prevent foreign fragment retrieval
    "fr", "de", "es", "pt", "ja", "zh", "ru", "ko", "it", "tr", "vi", 
    "id", "fa", "he", "ar", "uk", "nl", "pl", "hu", "cz", "sv"
}

def load_local_directory(root_path="."):
    """Scans project files and filters out non-English documentation folders."""
    documents = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        normalized_path = dirpath.replace("\\", "/")
        
        # Prune ignored folders from traversal
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_DIRS]
        
        # Strict filter: If inside a 'docs' directory, enforce English paths
        if "docs/" in normalized_path and "/en" not in normalized_path and not normalized_path.endswith("/docs"):
            continue

        for file in filenames:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                file_path = os.path.join(dirpath, file)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    if not content.strip():
                        continue
                        
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "filename": file,
                            "type": "code" if ext in {".py", ".js"} else "documentation"
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Skipping {file_path} due to read error: {e}")
                    
    return documents

def build_offline_index():
    print("Starting Offline Index Builder (English-Only Mode)...")

    # 1. INGESTION
    print("Scanning project directory for source and documentation files...")
    raw_docs = load_local_directory(".")
    
    if not raw_docs:
        print("Error: No valid documents found to index.")
        return
        
    print(f"Successfully loaded {len(raw_docs)} English source/doc files.")

    # 2. CHUNKING
    print("Chunking documents (size: 700, overlap: 100)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunked_docs)} semantic chunks.")

    # 3. EMBEDDINGS (BAAI/bge-small-en-v1.5)
    print("Loading HuggingFace embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. BUILD & SAVE FAISS INDEX
    print("Building FAISS vector index...")
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)

    export_dir = "data/faiss_index"
    os.makedirs(export_dir, exist_ok=True)
    vectorstore.save_local(export_dir)
    
    print(f"Success! Clean FAISS index saved to '{export_dir}'.")

if __name__ == "__main__":
    build_offline_index()