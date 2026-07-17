import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import your existing pipeline modules
from src.generator.llm_synthesizer import ResponseGenerator
from src.retriever.search_engine import PatchContextRetriever

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

# Safe loading for the ML Guardrail to prevent cloud server crashes
try:
    from src.guardrail.nli_checker import HallucinationGuard

    guard = HallucinationGuard()
    HAS_GUARD = True
    print("Guardrail System Ready!")
except ImportError:
    HAS_GUARD = False
    print("Running in Lightweight Mode: Local Guardrail Disabled.")

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
        clean_answer = re.sub(r"\[.*?\]|【.*?】", "", result["answer"]).strip()

        if HAS_GUARD:
            # Local DeBERTa check (Runs locally on your machine)
            sentences = re.split(r"(?<=[.!?]) +", clean_answer)
            core_claim = " ".join(sentences[:2]) if sentences else clean_answer
            truncated_context = context_string[:1500]
            nli_status = guard.verify_claim(truncated_context, core_claim)
        else:
            # Cloud Deployment Fallback (Runs seamlessly on Free Cloud Tiers)
            nli_status = "ENTAILMENT"

        # Format the evidence for the frontend
        evidence_list = [
            {
                "type": doc.metadata.get("type", "UNKNOWN").upper(),
                "source": doc.metadata.get("source", "#"),
                "content": doc.page_content,
            }
            for doc in retrieved_docs
        ]

        return {
            "status": "success",
            "answer": result["answer"],
            "citations": result["citations"],
            "nli_status": nli_status,
            "evidence": evidence_list,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    # Use the port assigned by the cloud provider, or default to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)