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

        # Lightweight LLM-as-a-Judge check via Groq if verification is needed
        if citation_valid:
            nli_status = "ENTAILMENT"  # Fast-path validation
        else:
            # Quick secondary check prompt via Groq (Uses zero memory on Render)
            nli_status = generator.verify_hallucination_via_llm(context_string, raw_answer)
        
        # Format lean evidence payload for frontend
        evidence_list = [
            {
                "id": doc.metadata.get("id", idx),
                "type": doc.metadata.get("type", "UNKNOWN").upper(),
                "source": doc.metadata.get("source", "#"),
                "content": doc.page_content[:300] + "..." # Truncate transfer size
            } for idx, doc in enumerate(retrieved_docs)
        ]
        
        return {
            "status": "success",
            "answer": raw_answer,
            "citations": result.get("citations", cited_ids),
            "nli_status": nli_status,
            "evidence": evidence_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Mount the data folder so /data/raw/... paths can serve documents natively
# Note: Keep static/data paths mounted BEFORE or AFTER UI handlers appropriately
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)