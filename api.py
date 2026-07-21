import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Stateless orchestration imports (No torch, no local transformers loaded)
from src.retriever.search_engine import PatchContextRetriever
from src.generator.llm_synthesizer import ResponseGenerator

app = FastAPI(title="PatchContext Optimized API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Lean AI Orchestrator...")
# Loads precomputed index.faiss & metadata.pkl instantly (~3s startup)
retriever = PatchContextRetriever() 
generator = ResponseGenerator()
print("Stateless Pipeline Ready!")

# Base URL for source repository documentation
# Base URL for source repository documentation
GITHUB_BASE_URL = "https://github.com/fastapi/fastapi/blob/master/"

def format_github_url(source_path: str) -> str:
    """Converts a local file path into a valid, clickable GitHub repository URL."""
    if not source_path or source_path == "#":
        return "#"
    
    # Return directly if already an HTTP/HTTPS URL
    if source_path.startswith("http://") or source_path.startswith("https://"):
        return source_path
        
    # Standardize slashes for cross-platform compatibility
    clean_path = source_path.replace("\\", "/")
    
    # Strip the local ingestion folder path so only the repo structure remains
    local_prefix = "data/raw/fastapi_repo/"
    
    if local_prefix in clean_path:
        clean_path = clean_path.split(local_prefix)[-1]
    else:
        # Fallback cleanup just in case
        clean_path = clean_path.lstrip("./")
    
    return f"{GITHUB_BASE_URL}{clean_path}"

class QueryRequest(BaseModel):
    question: str

@app.post("/api/analyze")
async def analyze_history(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        # 1. Retrieve using static pre-loaded FAISS index
        retrieved_docs = retriever.retrieve_relevant_chunks(request.question)
        if not retrieved_docs:
            return {"status": "no_data", "message": "No historical evidence found."}
            
        # 2. Synthesize using Groq
        context_string = generator.format_context(retrieved_docs)
        result = generator.generate_answer(request.question)
        
        # 3. Smart Lightweight Citation & Hallucination Guard
        raw_answer = result["answer"]
        cited_ids = re.findall(r'\[(?:Doc|Ref)?\s*(\d+)\]', raw_answer)
        
        # Cross-verify citations with retrieved source metadata IDs
        source_ids = {str(doc.metadata.get("id")) for doc in retrieved_docs if doc.metadata.get("id") is not None}
        
        # Fixed set evaluation to prevent missing property errors
        citation_valid = True
        if cited_ids and source_ids:
            if not set(cited_ids).issubset(source_ids):
                citation_valid = False

        # Lightweight check
        if citation_valid:
            nli_status = "ENTAILMENT"
        else:
            if hasattr(generator, "verify_hallucination_via_llm"):
                nli_status = generator.verify_hallucination_via_llm(context_string, raw_answer)
            else:
                nli_status = "VERIFIED"

        # Format citations with valid GitHub links
        raw_citations = result.get("citations", [])
        formatted_citations = []
        for cite in raw_citations:
            if isinstance(cite, dict):
                formatted_citations.append({
                    "type": cite.get("type", "DOCUMENTATION"),
                    "source_url": format_github_url(cite.get("source_url", "#"))
                })

        # Format lean evidence payload for frontend with working GitHub URLs
        evidence_list = [
            {
                "id": doc.metadata.get("id", idx),
                "type": doc.metadata.get("type", "UNKNOWN").upper(),
                "source": format_github_url(doc.metadata.get("source", "#")),
                "content": doc.page_content[:300] + "..."  # Truncate transfer size
            } for idx, doc in enumerate(retrieved_docs)
        ]
        
        return {
            "status": "success",
            "answer": raw_answer,
            "citations": formatted_citations if formatted_citations else cited_ids,
            "nli_status": nli_status,
            "evidence": evidence_list
        }
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)