import os
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RepositoryDataProcessor:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        # Using 500 characters/tokens for clean, digestible developer contexts
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_commits(self) -> list[Document]:
        """Converts raw commits JSON into structured LangChain Documents."""
        filepath = os.path.join(self.data_dir, "commits.json")
        documents = []
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found.")
            return documents
            
        with open(filepath, "r", encoding="utf-8") as f:
            commits = json.load(f)
            
        for c in commits:
            # Build an explicit, readable string format for the model to parse
            text_content = (
                f"Type: Commit\n"
                f"SHA: {c['sha']}\n"
                f"Author: {c['author']}\n"
                f"Date: {c['date']}\n"
                f"Message: {c['message']}"
            )
            
            # Metadata is critical for the generation and citation layer later
            metadata = {
                "type": "commit",
                "sha": c["sha"],
                "author": c["author"],
                "source": f"https://github.com/fastapi/fastapi/commit/{c['sha']}"
            }
            
            documents.append(Document(page_content=text_content, metadata=metadata))
            
        print(f"Processed {len(documents)} commits into documents.")
        return documents

    def load_issues_and_prs(self) -> list[Document]:
        """Converts raw issues and PRs JSON into structured LangChain Documents."""
        filepath = os.path.join(self.data_dir, "issues_prs.json")
        documents = []
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found.")
            return documents
            
        with open(filepath, "r", encoding="utf-8") as f:
            items = json.load(f)
            
        for item in items:
            text_content = (
                f"Type: {item['type'].upper()}\n"
                f"Number: #{item['number']}\n"
                f"Title: {item['title']}\n"
                f"State: {item['state']}\n"
                f"Created At: {item['created_at']}\n"
                f"Description: {item['body']}"
            )
            
            metadata = {
                "type": item["type"],
                "number": item["number"],
                "title": item["title"],
                "source": item["html_url"]
            }
            
            documents.append(Document(page_content=text_content, metadata=metadata))
            
        print(f"Processed {len(documents)} issues/PRs into documents.")
        return documents

    def process_and_chunk_repository(self) -> list[Document]:
        """Loads all repository resources and splits them into semantic chunks."""
        raw_docs = []
        raw_docs.extend(self.load_commits())
        raw_docs.extend(self.load_issues_and_prs())
        
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
    
    # Quick debug inspection of a chunk
    if chunks:
        print("\n--- Sample Chunk Preview ---")
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Content:\n{chunks[0].page_content}")