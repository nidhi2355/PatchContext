# PatchContext 🚀

A high-performance, ultra-lightweight, and completely stateless Retrieval-Augmented Generation (RAG) application optimized to run smoothly within highly constrained production environments (such as the Render Free Tier with a strict **512MB RAM** limit).

🌐 **Live Demo:** [https://patchcontext.onrender.com/](https://patchcontext.onrender.com/)


---


## 💡 The Architecture Challenge & Strategy

Traditional production RAG pipelines often suffer from intense memory footprints and long cold-start times because they load heavy Deep Learning frameworks (`torch`, `transformers`) and embedding models directly into the production server's memory. On 512MB environments, this instantly triggers an Out-Of-Memory (OOM) crash.

**PatchContext solves this by decoupling ML heavy-lifting from production:**

1. **Pre-computed Indexing (Local):** Raw documentation parsing, recursive text chunking, and FAISS vector database initialization are completely handled locally using a specialized build script.
2. **Stateless API Inference (Cloud):** The production FastAPI server hosts *only* the ultra-lightweight, pre-compiled FAISS index and handles text logic. 
3. **Serverless Vector Math:** Embedding generation is offloaded to the Hugging Face Cloud Inference API, and LLM synthesis is managed via Groq's hardware-accelerated API. 

This results in an application that consumes **< 150MB of RAM** under peak production loads and achieves sub-second query execution times.


---


## 🛠️ Tech Stack

*   **Backend Framework:** FastAPI (Python)
*   **LLM Inference Core:** Groq API (`llama-3.3-70b-versatile`)
*   **Vector Database Wrapper:** FAISS (CPU edition)
*   **Orchestration Engine:** LangChain Community & LangChain Hugging Face
*   **Cloud Embeddings Engine:** Hugging Face Inference API (`all-MiniLM-L6-v2`)
*   **Server Infrastructure:** Render (Stateless Web Service)


---


## 📂 Project Structure

```text
├── .github/               # CI/CD workflows
├── data/
│   └── raw/               # Markdown/Text source files containing knowledge base context
├── faiss_index/           # COMPILER INDEX: Pre-calculated vector index files (git-tracked)
│   ├── index.faiss
│   └── index.pkl
├── src/
│   ├── generator/         # LLM prompt composition and hallucination guard modules
│   └── retriever/         # Optimized FAISS search logic
├── static/                # Single-page frontend application assets
│   └── index.html         # Main UI console interface
├── api.py                 # Lean FastAPI server application
├── build_index.py         # Local data-compilation script (Never run on server)
└── requirements.txt       # Production dependencies (PyTorch & Transformers excluded)
```

---


## 🔧 Local Setup & Vector Index Compilation

To add new source materials or adjust the application dataset, update your vector files locally before pushing changes upstream.

### 1. Clone the Repository & Configure Virtual Environment
```bash
git clone [https://github.com/yourusername/patchcontext.git](https://github.com/yourusername/patchcontext.git)
cd patchcontext

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```



### 2. Configure Environment Secrets
Create a .env file in the root folder:

Code snippet
```bash
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_access_token_here
```




### 3. Add Raw Data and Compile Your Index
Place your unstructured reference documentation or source logs inside the data/raw/ directory. Run the indexing compiler to calculate embeddings and rebuild your vector maps:

```bash
python build_index.py
This automatically updates the binary structural records saved within the faiss_index/ bundle folder.
```




### 4. Test the Backend Locally
```bash
python api.py
Open your browser and navigate to http://localhost:8000 to test your query latency locally.
```


---


## 🚀 Cloud Deployment (Render Guidelines)
When configuring your Web Service instance inside the Render Console interface, apply these exact configuration definitions to avoid runtime issues:

1. **Runtime Environment**: Python

2. **Build Command**: pip install -r requirements.txt

3. **Start Command**: python api.py

4. **Environment Variables**: Add your active GROQ_API_KEY and HF_TOKEN configuration keys directly inside the service's Environment parameters tab.


---


## 🔒 Smart Hallucination Guard
PatchContext implements a dual-layer validation workflow to prevent LLM hallucinations:

1. **Rigorous Structural Validation**: Cross-references any LLM-generated citations ([Doc 1], [Ref 2]) directly against the unique indices of the documents retrieved by FAISS. If an invisible source signature is fabricated, it is caught instantly.

2. **Zero-Memory NLI Evaluation**: Automatically routes ambiguous responses through a targeted Cloud Natural Language Inference (NLI) verification routine via Groq to guarantee factual entailment without increasing server load.
