import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define which file types to index and which folders to ignore
ALLOWED_EXTENSIONS = {".py", ".md", ".txt", ".html", ".js", ".json"}
IGNORE_DIRS = {".git", "venv", "env", "__pycache__", "node_modules", "faiss_index"}

def load_local_directory(root_path="."):
    """Scans the local directory and loads files into LangChain Documents."""
    documents = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Remove ignored directories so we don't traverse them
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        
        for file in filenames:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                file_path = os.path.join(dirpath, file)
                
                try:
                    # Read the file content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    # Skip empty files
                    if not content.strip():
                        continue
                        
                    # Create a LangChain Document with metadata
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
    print("Starting Offline Index Builder (Real Data Mode)...")

    # 1. INGESTION: Scan local project files
    print("Scanning local directory for project files...")
    raw_docs = load_local_directory(".")
    
    if not raw_docs:
        print("Error: No valid files found to index. Check your ALLOWED_EXTENSIONS.")
        return
        
    print(f"Successfully loaded {len(raw_docs)} files from your project.")

    # 2. CHUNKING: Optimized for context window and fast retrieval
    print("Chunking documents (size: 700, overlap: 100)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunked_docs)} individual chunks.")

    # 3. EMBEDDING: Run locally
    print("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. VECTOR DATABASE: Build the FAISS index
    print("Building FAISS vector database. This may take a moment depending on your codebase size...")
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)

    # 5. EXPORT: Save the static files for Render
    export_dir = "faiss_index"
    vectorstore.save_local(export_dir)
    
    print(f"Success! Real data FAISS index saved to the '{export_dir}' directory.")
    print("You can now push this to GitHub and deploy!")

if __name__ == "__main__":
    build_offline_index()