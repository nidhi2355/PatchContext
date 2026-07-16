import os
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RepositoryDataProcessor:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.clone_dir = os.path.join(data_dir, "fastapi_repo")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,  # Slightly increased for better context retention
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_commits(self) -> list[Document]:
        """Converts raw commits JSON into structured LangChain Documents."""
        filepath = os.path.join(self.data_dir, "commits.json")
        documents = []
        if not os.path.exists(filepath):
            return documents
        with open(filepath, "r", encoding="utf-8") as f:
            commits = json.load(f)
        for c in commits:
            text_content = f"Type: Commit\nSHA: {c['sha']}\nAuthor: {c['author']}\nDate: {c['date']}\nMessage: {c['message']}"
            metadata = {"type": "commit", "sha": c["sha"], "source": f"https://github.com/fastapi/fastapi/commit/{c['sha']}"}
            documents.append(Document(page_content=text_content, metadata=metadata))
        return documents

    def load_issues_and_prs(self) -> list[Document]:
        """Converts raw issues and PRs JSON into structured LangChain Documents."""
        filepath = os.path.join(self.data_dir, "issues_prs.json")
        documents = []
        if not os.path.exists(filepath):
            return documents
        with open(filepath, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            text_content = f"Type: {item['type'].upper()}\nNumber: #{item['number']}\nTitle: {item['title']}\nDescription: {item['body']}"
            metadata = {"type": item["type"], "number": item["number"], "source": item["html_url"]}
            documents.append(Document(page_content=text_content, metadata=metadata))
        return documents

    def load_documentation_files(self) -> list[Document]:
        """Scans the cloned repository for markdown documentation files and indexes them."""
        documents = []
        docs_path = os.path.join(self.clone_dir, "docs", "en", "docs") # Focus on English docs
        
        if not os.path.exists(docs_path):
            # Fallback to checking the entire clone directory for any .md files if specific structure differs
            docs_path = self.clone_dir
            
        if not os.path.exists(docs_path):
            print(f"Warning: Cloned repository directory not found at {docs_path}")
            return documents

        print(f"Scanning for markdown documentation inside: {docs_path}...")
        for root, _, files in os.walk(docs_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        # Calculate a clean display path relative to the repo root
                        relative_path = os.path.relpath(file_path, self.clone_dir)
                        
                        text_content = f"Type: Documentation\nFile: {relative_path}\nContent:\n{content}"
                        metadata = {
                            "type": "documentation",
                            "source": f"https://github.com/fastapi/fastapi/blob/master/{relative_path.replace(os.sep, '/')}"
                        }
                        documents.append(Document(page_content=text_content, metadata=metadata))
                    except Exception as e:
                        print(f"Skipping file {file_path} due to read error: {e}")
                        
        print(f"Processed {len(documents)} markdown documentation files into documents.")
        return documents

    def process_and_chunk_repository(self) -> list[Document]:
        """Loads all repository resources including documentation and splits them into semantic chunks."""
        raw_docs = []
        raw_docs.extend(self.load_commits())
        raw_docs.extend(self.load_issues_and_prs())
        raw_docs.extend(self.load_documentation_files())  # Core integration fix
        
        if not raw_docs:
            print("No documents found to chunk.")
            return []
            
        print("Splitting documents into semantic chunks...")
        chunks = self.text_splitter.split_documents(raw_docs)
        print(f"Total semantic chunks generated: {len(chunks)}")
        return chunks

if __name__ == "__main__":
    processor = RepositoryDataProcessor()
    chunks = processor.process_and_chunk_repository()