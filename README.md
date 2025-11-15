# RAG Agent with LangGraph (plan → retrieve → answer → reflect)
- LangChain for document loading, chunking, embeddings and Chroma vector store
- HuggingFace sentence-transformers embeddings (all-MiniLM-L6-v2)
- OpenAI or HuggingFace endpoint LLM for generation and reflection
- Streamlit UI for interactive Q&A (bonus)


## Features
- Four LangGraph nodes: `plan`, `retrieve`, `answer`, `reflect`.
- Minimal logging for each node for easy line-by-line walkthrough.
- Optional tracing via LangSmith or TruLens (if API keys & packages are available).
- Evaluation utilities: ROUGE / BERTScore wrappers + an LLM judge method.


## Setup
1. Create virtual environment & install requirements:
```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
2. Set environment variables:
- Use OpenAI: `export OPENAI_API_KEY="sk-..."`
- Or HuggingFace endpoint: `export HUGGINGFACEHUB_API_TOKEN="hf_..."` and `export HUGGINGFACE_LLM_REPO_ID="your-hf-endpoint"`
- (Optional) LangSmith: `export LANGSMITH_API_KEY="..."`


3. Run Streamlit UI:
```bash
streamlit run rag_agent_with_langgraph.py
```


## Files
- `rag_agent_with_langgraph.py` — this file: contains code, small README and requirements strings.


## How it works (brief)
- `plan` node inspects the question and decides whether retrieval is necessary.
- `retrieve` node loads the PDF, splits text, builds embeddings and a Chroma vector store, and fetches top-k chunks.
- `answer` node formats a prompt using retrieved context and calls the LLM to generate an answer.
- `reflect` node asks an LLM to evaluate relevance/completeness and returns a numeric rating + reason.


