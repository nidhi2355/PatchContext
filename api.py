from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import re

# Import your existing pipeline modules
from src.retriever.search_engine import PatchContextRetriever
from src.generator.llm_synthesizer import ResponseGenerator
from src.guardrail.nli_checker import HallucinationGuard

app = FastAPI(title="PatchContext API")

# Essential: Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG components once on startup
print("Initializing AI Pipeline...")
retriever = PatchContextRetriever()
generator = ResponseGenerator()
guard = HallucinationGuard()
print("Pipeline Ready!")

# Define the data structure for incoming requests
class QueryRequest(BaseModel):
    question: str

@app.post("/api/analyze")
async def analyze_history(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        # 1. Retrieve
        retrieved_docs = retriever.retrieve_relevant_chunks(request.question)
        if not retrieved_docs:
            return {"status": "no_data", "message": "No historical evidence found."}
            
        # 2. Generate
        context_string = generator.format_context(retrieved_docs)
        result = generator.generate_answer(request.question)
        
        # 3. Guardrail Fact Check Optimization
        import re
        clean_answer = re.sub(r'\[.*?\]|【.*?】', '', result["answer"]).strip()
        
        # MNLI Fix: Extract only the first 1-2 sentences (the core claim) to verify.
        # This prevents the NLI model from failing on long, multi-paragraph essays.
        sentences = re.split(r'(?<=[.!?]) +', clean_answer)
        core_claim = " ".join(sentences[:2]) if sentences else clean_answer
        
        # Truncate context to ~1500 chars to avoid DeBERTa's 512 token limit
        truncated_context = context_string[:1500] 
        
        nli_status = guard.verify_claim(truncated_context, core_claim)
        
        # Format the evidence for the frontend
        evidence_list = [
            {
                "type": doc.metadata.get("type", "UNKNOWN").upper(),
                "source": doc.metadata.get("source", "#"),
                "content": doc.page_content
            } for doc in retrieved_docs
        ]
        
        return {
            "status": "success",
            "answer": result["answer"],
            "citations": result["citations"],
            "nli_status": nli_status,
            "evidence": evidence_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

@app.post("/api/analyze")
async def analyze_history(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        # 1. Retrieve
        retrieved_docs = retriever.retrieve_relevant_chunks(request.question)
        if not retrieved_docs:
            return {"status": "no_data", "message": "No historical evidence found."}
            
        # 2. Generate
        context_string = generator.format_context(retrieved_docs)
        result = generator.generate_answer(request.question)
        
        # 3. Guardrail Fact Check Optimization
        # Strip citation markers like [Evidence 1] or 【Evidence 1】 from the answer before checking
        clean_answer = re.sub(r'\[.*?\]|【.*?】', '', result["answer"]).strip()
        
        # To avoid DeBERTa 512-token truncation, we limit the context string length being checked
        # 2000 chars roughly equals ~400 tokens, leaving enough room for the generated answer
        truncated_context = context_string[:2000] 
        
        nli_status = guard.verify_claim(truncated_context, clean_answer)
        
        # Format the evidence for the frontend
        evidence_list = [
            {
                "type": doc.metadata.get("type", "UNKNOWN").upper(),
                "source": doc.metadata.get("source", "#"),
                "content": doc.page_content
            } for doc in retrieved_docs
        ]
        
        return {
            "status": "success",
            "answer": result["answer"],
            "citations": result["citations"],
            "nli_status": nli_status,
            "evidence": evidence_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))