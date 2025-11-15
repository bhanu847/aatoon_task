import os
import tempfile
import shutil
import traceback
from typing import Any, Dict, List
import streamlit as st

try:
    from langgraph.graph import StateGraph, START, END
except Exception as e:
    raise RuntimeError("langgraph is required. Install with `pip install langgraph`.") from e

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except Exception:
    ROUGE_AVAILABLE = False

try:
    import bert_score
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False
TRULENS_AVAILABLE = False
LANGSMITH_AVAILABLE = False
try:
    import trulens_eval
n  
    TRULENS_AVAILABLE = True
except Exception:
    TRULENS_AVAILABLE = False

try:
    import langsmith
    LANGSMITH_AVAILABLE = True
except Exception:
    LANGSMITH_AVAILABLE = False

UPLOAD_DIR = tempfile.gettempdir()
CHROMA_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "chroma_langgraph_streamlit")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVE_K = 4

ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use ONLY the context to answer the question.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\n"
        "Answer clearly and cite page numbers when possible. If the answer is not in the context, say so."
    )
)

REFLECT_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "context_snippets"],
    template=(
        "You are an evaluator. Given the QUESTION, the ANSWER produced, and the retrieved CONTEXT_SNIPPETS, "
        "rate whether the ANSWER is relevant and complete on a scale from 1 to 5 (5 best). "
        "Return a JSON object with fields: rating (int 1-5) and reason (one-sentence).\n\n"
        "QUESTION: {question}\n\nANSWER: {answer}\n\nCONTEXT_SNIPPETS:\n{context_snippets}\n\n"
    )
)


def log(title: str, payload: Any):
    print("\n" + "=" * 30)
    print(f"[{title}]")
    try:
        if isinstance(payload, (list, tuple)):
            for i, p in enumerate(payload[:5]):
                print(f"[{i}] {str(p)[:500]}")
            if len(payload) > 5:
                print(f"... ({len(payload)} items total)")
        else:
            print(str(payload)[:1000])
    except Exception as e:
        print("Log error:", e)
    print("=" * 30 + "\n")


def get_llm_callable():
    """Return a callable LLM. Prefers OpenAI, falls back to HuggingFace endpoint if configured."""
    if os.environ.get("OPENAI_API_KEY"):
        log("llm_select", "Using OpenAI LLM (OPENAI_API_KEY present)")
        return OpenAI(temperature=0)
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    hf_repo = os.environ.get("HUGGINGFACE_LLM_REPO_ID")
    if hf_token and hf_repo and HF_AVAILABLE:
        log("llm_select", f"Using HuggingFace endpoint LLM: {hf_repo}")
        raw = HuggingFaceEndpoint(repo_id=hf_repo, task="text-generation")
        return ChatHuggingFace(llm=raw)
    raise RuntimeError("No LLM configured. Set OPENAI_API_KEY or HUGGINGFACEHUB_API_TOKEN + HUGGINGFACE_LLM_REPO_ID")
  
from typing import TypedDict

class AgentState(TypedDict, total=False):
    question: str
    temp_pdf_path: str
    plan: dict
    retrieved_docs: list
    retrieved_snippets: list
    answer: str
    reflect: dict

def node_plan(state: AgentState) -> AgentState:
    q = (state.get("question") or "").strip()
    keywords = ["what", "how", "explain", "list", "benefits", "advantages", "summary"]
    need = any(k in q.lower() for k in keywords) and len(q.split()) >= 2
    plan = {"question": q, "need_retrieval": bool(need), "retrieval_k": RETRIEVE_K}
    log("plan", plan)
    return {"plan": plan}

def node_retrieve(state: AgentState) -> AgentState:
    plan = state.get("plan") or {}
    if not plan.get("need_retrieval"):
        log("retrieve", "Retrieval not needed per plan")
        return {"retrieved_docs": [], "retrieved_snippets": []}

    pdf_path = state.get("temp_pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF not found: " + str(pdf_path))

    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    log("retrieve:loaded_docs_count", len(raw_docs))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)
    log("retrieve:chunks_count", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # Clear previous chroma for isolation
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception:
            pass

    vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
    vs.persist()
    retriever = vs.as_retriever(search_kwargs={"k": plan.get("retrieval_k", RETRIEVE_K)})
    docs_found = retriever.get_relevant_documents(plan["question"])
    log("retrieve:found_count", len(docs_found))

    snippets = []
    for d in docs_found:
        text = d.page_content.replace("\n", " ")
        page_info = d.metadata.get("page") if isinstance(d.metadata, dict) else None
        if page_info is not None:
            snippets.append(f"[page {page_info}] {text[:400]}")
        else:
            snippets.append(text[:400])

    return {"retrieved_docs": docs_found, "retrieved_snippets": snippets}


def node_answer(state: AgentState) -> AgentState:
    plan = state.get("plan") or {}
    question = plan.get("question") or state.get("question") or ""
    docs = state.get("retrieved_docs") or []
    llm = get_llm_callable()

    if not docs:
        prompt = f"No context. Answer concisely or say you can't answer from the PDF.\nQuestion: {question}"
        log("answer:prompt_no_context", prompt[:800])
        res = llm(prompt)
        ans = getattr(res, "content", str(res))
        log("answer", ans[:800])
        return {"answer": ans}

    context_parts = []
    for d in docs[:6]:
        p = d.metadata.get("page") if isinstance(d.metadata, dict) else None
        header = f"[page {p}] " if p is not None else ""
        context_parts.append(header + d.page_content)

    context = "\n\n---\n\n".join(context_parts)
    formatted = ANSWER_PROMPT.format(context=context, question=question)
    log("answer:formatted_prompt", formatted[:1200])
    resp = llm(formatted)
    ans_text = getattr(resp, "content", str(resp))
    log("answer:result", ans_text[:1000])
    return {"answer": ans_text}

def node_reflect(state: AgentState) -> AgentState:
    question = state.get("plan", {}).get("question") or state.get("question") or ""
    answer = state.get("answer") or ""
    snippets = state.get("retrieved_snippets") or []

    joined = "\n\n---\n\n".join(snippets[:6]) if snippets else "No snippets"
    eval_prompt = REFLECT_PROMPT.format(question=question, answer=answer, context_snippets=joined)
    log("reflect:prompt", eval_prompt[:1200])
    llm = get_llm_callable()
    ev = llm(eval_prompt)
    ev_text = getattr(ev, "content", str(ev))
    log("reflect:raw", ev_text[:800])

    rating = 3
    reason = ev_text.strip().replace("\n", " ")[:300]
    try:
        import json, re
        cand = ev_text.strip()
        if cand.startswith("{"):
            parsed = json.loads(cand)
            rating = int(parsed.get("rating", rating))
            reason = parsed.get("reason", reason)
        else:
            m = re.search(r"[1-5]", ev_text)
            if m:
                rating = int(m.group())
    except Exception:
        pass

    reflect = {"rating": max(1, min(5, int(rating))), "reason": reason}
    log("reflect:final", reflect)
    return {"reflect": reflect}


workflow = StateGraph(AgentState)
workflow.add_node("plan", node_plan)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("answer", node_answer)
workflow.add_node("reflect", node_reflect)
workflow.add_edge(START, "plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", "reflect")
workflow.add_edge("reflect", END)
compiled = workflow.compile()


def eval_rouge(pred: str, ref: str) -> Dict[str, float]:
    if not ROUGE_AVAILABLE:
        return {"rouge_l": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {k: float(v.fmeasure) for k, v in scores.items()}


def eval_bertscore(preds: List[str], refs: List[str]) -> Dict[str, float]:
    if not BERTSCORE_AVAILABLE:
        return {"bert_f1": 0.0}
    P, R, F1 = bert_score.score(preds, refs, lang="en", rescale_with_baseline=True)
    return {"bert_f1": float(F1.mean())}


def simple_overlap_metric(pred: str, ref: str) -> float:
    ps = set(pred.lower().split())
    rs = set(ref.lower().split())
    if not rs:
        return 0.0
    return len(ps & rs) / len(rs)

# LLM Judge wrapper (use LLM to compare pred vs ref)

def llm_judge(pred: str, ref: str) -> Dict[str, Any]:
    llm = get_llm_callable()
    prompt = ((
        "You are an evaluator. Given the reference answer and the model-predicted answer, "
        "rate the predicted answer between 1 and 5 for correctness and completeness and give a one-sentence reason.\n\n"
        f"REFERENCE:\n{ref}\n\nPREDICTION:\n{pred}\n\nRespond with JSON: {""}"rating":int, "reason":string{""}"  
   )
        
    resp = llm(prompt)
    txt = getattr(resp, "content", str(resp))
    
    try:
        import json, re
        if txt.strip().startswith("{"):
            parsed = json.loads(txt)
            return {"rating": int(parsed.get("rating", 3)), "reason": parsed.get("reason", "")} 
        m = re.search(r"[1-5]", txt)
        rating = int(m.group()) if m else 3
        return {"rating": rating, "reason": txt.strip()[:300]}
    except Exception:
        return {"rating": 3, "reason": txt.strip()[:300]}

# Streamlit UI ---------------------------------------

st.set_page_config(page_title="LangGraph RAG Agent", layout="centered")
st.title("LangGraph RAG Agent — plan → retrieve → answer → reflect")

st.markdown("Upload a PDF and ask questions. The graph executes for each request and prints logs to the server.")

uploaded = st.file_uploader("Upload PDF", type=["pdf"]) 
question = st.text_area("Question", value="What are the benefits of renewable energy?")
run_btn = st.button("Ask")

if run_btn:
    if not uploaded:
        st.error("Please upload a PDF first.")
    else:
        fname = uploaded.name
        temp_path = os.path.join(UPLOAD_DIR, fname)
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        try:
            init_state = {"question": question, "temp_pdf_path": temp_path}
            st.info("Running LangGraph workflow...")
            final = compiled.invoke(init_state)

            plan = final.get("plan", {})
            snippets = final.get("retrieved_snippets", [])
            answer = final.get("answer", "")
            reflect = final.get("reflect", {"rating": 3, "reason": ""})

            st.subheader("Plan")
            st.json(plan)

            st.subheader(f"Retrieved snippets (count {len(snippets)})")
            for i, s in enumerate(snippets):
                st.write(f"[{i+1}] {s}")

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Reflect")
            st.write(reflect)

            ref_text = st.text_area("(Optional) Reference / expected answer (for evaluation)")
            if ref_text:
                st.markdown("**Evaluation:**")
                rouge_scores = eval_rouge(answer, ref_text)
                bert = eval_bertscore([answer], [ref_text]) if BERTSCORE_AVAILABLE else {"bert_f1": 0.0}
                overlap = simple_overlap_metric(answer, ref_text)
                st.write({"rouge": rouge_scores, "bertscore": bert, "overlap": overlap})

                judge = llm_judge(answer, ref_text)
                st.write({"llm_judge": judge})

            if LANGSMITH_AVAILABLE and os.environ.get("LANGSMITH_API_KEY"):
                st.success("LangSmith tracing requested — but this demo only shows placeholder integration.")
            if TRULENS_AVAILABLE:
                st.success("TruLens detected — placeholder for adding model eval traces.")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass


print("Module loaded. Run 'streamlit run rag_agent_with_langgraph.py' to start the UI.")
