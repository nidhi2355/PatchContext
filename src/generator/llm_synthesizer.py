import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.retriever.search_engine import PatchContextRetriever

# Load the GEMINI_API_KEY from .env
load_dotenv()

class ResponseGenerator:
    def __init__(self):
        print("Initializing Groq GPT-OSS-120B...")
        # Initialize Groq using LangChain
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b", 
            temperature=0.2, 
            max_tokens=1024,
            max_retries=2
        )
        
        # Initialize our retrieval engine from Step 5
        self.retriever = PatchContextRetriever()
        
        # Define the strict system prompt for context-aware generation
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are PatchContext, an expert AI developer assistant. 
Your job is to explain the historical reasoning behind software development decisions.
You must answer the user's question based SOLELY on the retrieved repository evidence provided below.
If the evidence does not contain the answer, politely state that the repository history does not contain this information.
Do NOT make unsupported assumptions. Clearly explain the development rationale.

Retrieved Repository Evidence:
{context}
"""),
            ("human", "{question}")
        ])
        
        # Chain the prompt and the LLM together
        self.chain = self.prompt | self.llm

    def format_context(self, docs):
        """Formats the retrieved LangChain documents into a single readable string for the prompt."""
        formatted_text = ""
        for i, doc in enumerate(docs):
            formatted_text += f"\n--- Evidence {i+1} ---\n"
            formatted_text += f"Type: {doc.metadata.get('type', 'Unknown').upper()}\n"
            formatted_text += f"Source URL: {doc.metadata.get('source', 'Unknown')}\n"
            formatted_text += f"Content: {doc.page_content}\n"
        return formatted_text

    def generate_answer(self, query: str):
        """Retrieves context, generates an answer, and attaches citations."""
        
        # 1. Retrieve the best historical evidence
        print(f"\nSearching repository history for: '{query}'")
        retrieved_docs = self.retriever.retrieve_relevant_chunks(query)
        
        if not retrieved_docs:
            return {
                "answer": "I could not find any relevant commits, PRs, or issues in the repository history to answer this question.",
                "citations": []
            }
            
        # 2. Format the retrieved context
        context_string = self.format_context(retrieved_docs)
        
        # 3. Generate the response using Gemini Flash
        print("Generating synthesized explanation via Gemini Flash...")
        response = self.chain.invoke({
            "context": context_string,
            "question": query
        })
        
        # 4. Extract citations for the frontend UI
        citations = [
            {
                "type": doc.metadata.get("type", "unknown").upper(),
                "source_url": doc.metadata.get("source", "#")
            } 
            for doc in retrieved_docs
        ]
        
        # Remove duplicate citations while preserving order
        unique_citations = []
        seen_urls = set()
        for c in citations:
            if c["source_url"] not in seen_urls:
                unique_citations.append(c)
                seen_urls.add(c["source_url"])
                
        return {
            "answer": response.content,
            "citations": unique_citations
        }

if __name__ == "__main__":
    # Test the complete RAG generation pipeline
    generator = ResponseGenerator()
    test_query = "Why did the project introduce the JSONResponse class?"
    
    result = generator.generate_answer(test_query)
    
    print("\n================ FINAL GENERATED ANSWER ================\n")
    print(result["answer"])
    print("\n====================== CITATIONS =======================\n")
    for i, cite in enumerate(result["citations"]):
        print(f"[{i+1}] {cite['type']}: {cite['source_url']}")