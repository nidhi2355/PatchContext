import streamlit as st
from src.retriever.search_engine import PatchContextRetriever
from src.generator.llm_synthesizer import ResponseGenerator
from src.guardrail.nli_checker import HallucinationGuard

# Set up page configurations
st.set_page_config(
    page_title="PatchContext | AI Git History Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize our core components and cache them so they don't reload on every click
@st.cache_resource
def initialize_pipeline():
    retriever = PatchContextRetriever()
    generator = ResponseGenerator()
    guard = HallucinationGuard()
    return retriever, generator, guard

try:
    retriever, generator, guard = initialize_pipeline()
except Exception as e:
    st.error(f"Initialization Error: Ensure your vector index is built and .env is configured. Details: {e}")
    st.stop()

# App Title & Header Description
st.title("🤖 PatchContext")
st.subheader("AI-Powered Development History Assistant for FastAPI")
st.markdown(
    "Ask natural language questions about architectural shifts, design rationale, "
    "or deprecation decisions buried inside Git history, Pull Requests, and Issues."
)
st.markdown("---")

# Layout: Sidebar for quick links / project details
with st.sidebar:
    st.header("Project Insights")
    st.markdown("**Knowledge Base:** FastAPI Repository")
    st.markdown("**Embedding Engine:** BAAI/bge-small-en-v1.5 (Local)")
    st.markdown("**Reranker:** Cross-Encoder MS-MARCO (Local)")
    st.markdown("**LLM:** Google Gemini 3.5 Flash")
    st.markdown("**Guardrail:** DeBERTa-v3-MNLI (Local)")
    st.markdown("---")
    st.markdown("Created for rapid repository onboarding and historical verification.")

# User Input Form
query = st.text_input(
    "Enter your historical codebase question:",
    placeholder="e.g., Why did the project introduce the JSONResponse class?"
)

if st.button("Analyze Codebase History", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid question first.")
    else:
        with st.spinner("Searching repository archives and verifying facts..."):
            # 1. Step-by-Step Execution: Retrieval (Runs locally, always safe!)
            retrieved_docs = retriever.retrieve_relevant_chunks(query)
            
            if not retrieved_docs:
                st.error("No historical evidence found in the local vector database matching that query.")
            else:
                # 2. Step-by-Step Execution: Context Formatting & Generation
                context_string = generator.format_context(retrieved_docs)
                
                try:
                    # Safely attempt to generate response from Gemini
                    result = generator.generate_answer(query)
                    generated_answer = result["answer"]
                    citations = result["citations"]
                    
                    # 3. Step-by-Step Execution: Run the NLI Guardrail Verification
                    nli_status = guard.verify_claim(context_string, generated_answer)
                    
                    # Create Columns to separate Answer and Evidence visually
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.header("💡 Synthesized Explanation")
                        
                        # Display NLI Guardrail Badge
                        if nli_status == "ENTAILMENT":
                            st.success("🔒 Fact Check: Verified (Fully supported by repository history)")
                        elif nli_status == "NEUTRAL":
                            st.warning("⚠️ Fact Check: Caution (Contains claims not fully explicit in the text chunks)")
                        else:
                            st.error("🚨 Fact Check: Contradiction Detected (Answer conflicts with retrieved evidence)")
                        
                        # Output the core response text
                        st.markdown(generated_answer)
                        
                        # Render Citations gracefully
                        if citations:
                            st.markdown("### 📌 Verified Citations")
                            for idx, cite in enumerate(citations):
                                st.markdown(f"**[{idx+1}]** [{cite['type']}]({cite['source_url']}) — View original resource on GitHub")
                    
                    with col2:
                        st.header("📁 Retrieved Historical Evidence")
                        st.markdown("*The raw records mined by MMR & Reranked by the Cross-Encoder:*")
                        
                        for idx, doc in enumerate(retrieved_docs):
                            with st.expander(f"Evidence Fragment {idx+1} ({doc.metadata.get('type').upper()})"):
                                st.caption(f"Source URL: {doc.metadata.get('source')}")
                                st.code(doc.page_content, language="markdown")
                                
                except Exception as api_error:
                    # Catch temporary API spikes gracefully without crashing the frontend layout
                    st.error("🤖 Gemini API is experiencing high demand right now.")
                    st.warning("The local retrieval engine successfully found the data, but the remote LLM generator is busy. Please wait a few seconds and try clicking 'Analyze' again!")
                    
                    # Still display the retrieved evidence fragments so the developer isn't left empty-handed
                    st.markdown("---")
                    st.subheader("📁 Preview Local Retrieved Evidence fragments anyway:")
                    for idx, doc in enumerate(retrieved_docs):
                        with st.expander(f"Evidence Fragment {idx+1} ({doc.metadata.get('type').upper()})"):
                            st.code(doc.page_content, language="markdown")