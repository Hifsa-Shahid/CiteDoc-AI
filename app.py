import html, json, os, re, time
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="CiteDocs AI", page_icon="📚", layout="centered")

# ---------- settings ----------
GITHUB_URL = ""   # e.g. "https://github.com/Hifsa-Shahid/citedocs-ai"
AUTHOR = ""       # e.g. "Hifsa Shahid"


def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


GROQ_KEY = secret("GROQ_API_KEY")
MODEL = secret("LLM_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = "openai/gpt-oss-20b"
DENSE_THRESHOLD = float(secret("DENSE_THRESHOLD", 0.41))  # calibrated in the notebook (section 9b)
MAX_QUESTIONS = 15  # per visitor session, protects the free API quota

SYSTEM = (
    "You are a documentation assistant for FastAPI. Answer ONLY using the numbered sources provided. "
    "The question must be about FastAPI itself. If it is about another framework or library "
    "(Flask, Django, Node.js, pandas, etc.), about general programming, or if the sources do not "
    "directly answer it, reply with exactly: NOT_FOUND. "
    "Otherwise cite sources inline like [1] or [2] and include a short code example if the sources contain one."
)

EXAMPLES = {
    "🔐 Protect an endpoint with JWT": "How do I protect an endpoint with a JWT token?",
    "📁 Upload files": "How do I upload files?",
    "🐳 Deploy with Docker": "How do I deploy with Docker?",
    "⏱️ Run background tasks": "How do I run background tasks?",
    "🌐 Enable CORS": "How do I enable CORS?",
    "🚫 Off-topic test": "How do I bake a chocolate cake?",
}

# ---------- styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 820px;}
.hero {text-align:center; margin-bottom: 0.4rem;}
.hero h1 {font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -0.5px;}
.grad {background: linear-gradient(90deg, #6366F1, #06B6D4); -webkit-background-clip: text;
       background-clip: text; color: transparent;}
.hero p {font-size: 1.05rem; opacity: 0.8; margin: 0.4rem 0 0.8rem 0;}
.pills {display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-bottom: 1.2rem;}
.pills span {border: 1px solid rgba(128,128,128,0.35); border-radius: 999px; padding: 3px 12px;
             font-size: 0.8rem; background: rgba(128,128,128,0.08);}
div.stButton > button {width: 100%; border-radius: 10px; font-size: 0.88rem;}
.src-card {border: 1px solid rgba(128,128,128,0.3); border-radius: 12px; padding: 12px 14px;
           background: rgba(128,128,128,0.07); min-height: 128px;}
.src-n {display:inline-block; background:#4F46E5; color:white; border-radius:999px; width:22px;
        height:22px; line-height:22px; text-align:center; font-size:0.75rem; margin-right:6px;}
.src-title {font-weight: 600; font-size: 0.92rem;}
.bar {height:6px; border-radius:99px; background: rgba(128,128,128,0.25); margin: 10px 0 4px 0;}
.bar > div {height:100%; border-radius:99px; background: linear-gradient(90deg, #6366F1, #06B6D4);}
.src-card small {opacity: 0.7;}
.src-card a {font-size: 0.85rem; text-decoration: none;}
.tip {border: 1px dashed rgba(128,128,128,0.4); border-radius: 12px; padding: 14px 18px;
      text-align:center; opacity: 0.85; margin-top: 8px;}
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------- retrieval + generation ----------
def tok(s):
    return re.findall(r"\w+", s.lower())


@st.cache_resource(show_spinner="Loading documentation index...")
def load_data():
    chunks = json.load(open("chunks.json", encoding="utf-8"))
    vecs = np.array([c["vec"] for c in chunks], dtype="float32")
    bm25 = BM25Okapi([tok(c["title"] + " " + c["text"]) for c in chunks])
    return chunks, vecs, bm25


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def retrieve(q, k=3):
    chunks, vecs, bm25 = load_data()
    embedder = load_embedder()
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")[0]
    dense = vecs @ qv
    dense_rank = [int(i) for i in np.argsort(-dense)[:50]]
    sparse_rank = [int(i) for i in np.argsort(-bm25.get_scores(tok(q)))[:50]]
    fused = {}
    for ranked in (dense_rank, sparse_rank):  # reciprocal rank fusion
        for r, i in enumerate(ranked):
            fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)
    top = [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])][:k]
    return top, dense, float(dense[dense_rank[0]])


def llm_call(messages):
    from groq import Groq, RateLimitError
    client = Groq(api_key=GROQ_KEY)
    model = MODEL
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.1, max_tokens=1500,
                reasoning_effort="low", messages=messages)
            return (r.choices[0].message.content or "").strip()
        except RateLimitError:
            if attempt == 1:
                model = FALLBACK_MODEL
            time.sleep(5 * (attempt + 1))
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def run(q):
    chunks, _, _ = load_data()
    t0 = time.time()
    top, dense, best = retrieve(q)
    t_ret = time.time() - t0
    sources = [{"n": n + 1, "title": chunks[i]["title"], "url": chunks[i]["url"],
                "score": round(float(dense[i]), 2)} for n, i in enumerate(top)]
    if best < DENSE_THRESHOLD:
        return {"answer": None, "sources": [], "t_ret": t_ret, "t_llm": 0.0,
                "refused": "This question doesn't look related to the FastAPI documentation."}
    context = "\n\n".join(f"[{s['n']}] ({chunks[i]['title']})\n{chunks[i]['text']}"
                          for s, i in zip(sources, top))
    t1 = time.time()
    text = llm_call([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {q}"}])
    t_llm = time.time() - t1
    if text is None:
        return {"answer": None, "sources": sources, "t_ret": t_ret, "t_llm": t_llm, "no_cache": True,
                "refused": "The AI service is busy right now. Please try again in a minute."}
    if not text or "NOT_FOUND" in text:
        return {"answer": None, "sources": sources, "t_ret": t_ret, "t_llm": t_llm,
                "refused": "The retrieved documentation doesn't answer this question."}
    return {"answer": text, "refused": None, "sources": sources, "t_ret": t_ret, "t_llm": t_llm}


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 📚 CiteDocs AI")
    st.caption("Unofficial. Not affiliated with FastAPI.")
    st.markdown("### How it works")
    st.markdown(
        "1. **Split** the docs by heading\n"
        "2. **Match** your question by meaning *and* keywords\n"
        "3. **Merge** both rankings (hybrid search)\n"
        "4. **Answer** with an LLM using only the top pages, with citations\n"
        "5. **Refuse** when the docs don't cover the question"
    )
    if os.path.exists("results.json"):
        res = json.load(open("results.json"))
        st.markdown("### Evaluation")
        st.caption(f"{res.get('n_test_questions', '?')} hand-checked questions · "
                   f"{res.get('n_chunks', '?')} doc chunks")
        chart = pd.DataFrame({"First-result accuracy (%)":
                              {k: round(v["hit@1"] * 100) for k, v in res["retrieval"].items()}})
        st.bar_chart(chart)
        tbl = pd.DataFrame(res["retrieval"]).T.rename(
            columns={"hit@1": "Top-1", "hit@3": "Top-3", "hit@5": "Top-5"})
        for c in ["Top-1", "Top-3", "Top-5"]:
            tbl[c] = (tbl[c] * 100).round(0).astype(int).astype(str) + "%"
        tbl["MRR"] = tbl["MRR"].round(2)
        st.dataframe(tbl)
        st.caption("Share of questions where the right docs page ranked in the top N. "
                   "Measured in Colab; this live demo uses hybrid retrieval without the reranker.")
    if GITHUB_URL:
        st.markdown(f"[⭐ View code on GitHub]({GITHUB_URL})")

# ---------- main page ----------
st.markdown("""
<div class="hero">
  <h1><span class="grad">CiteDocs AI</span></h1>
  <p>Ask about FastAPI in plain English. Get an answer with citations, straight from the official docs.</p>
</div>
<div class="pills">
  <span>🔎 Hybrid search</span><span>📎 Cited answers</span><span>🛑 Says "not found" instead of guessing</span>
</div>
""", unsafe_allow_html=True)


def set_q(text):
    st.session_state["q"] = text


labels = list(EXAMPLES.keys())
for row in range(0, len(labels), 3):
    cols = st.columns(3)
    for col, label in zip(cols, labels[row:row + 3]):
        col.button(label, on_click=set_q, args=(EXAMPLES[label],), key=f"ex_{label}")

q = st.text_input("Your question", key="q", label_visibility="collapsed",
                  placeholder="Type your own question, e.g. How do I protect an endpoint with a password?")

if "asked" not in st.session_state:
    st.session_state["asked"] = set()

if not q.strip():
    st.markdown('<div class="tip">👆 Pick an example above or type your own question.<br>'
                '<small>Try the off-topic button to see it refuse instead of guessing.</small></div>',
                unsafe_allow_html=True)
elif q not in st.session_state["asked"] and len(st.session_state["asked"]) >= MAX_QUESTIONS:
    st.warning("Question limit reached for this session. Refresh the page to continue.")
elif not GROQ_KEY:
    st.error("GROQ_API_KEY is not set. Add it in the app's Secrets settings.")
else:
    with st.spinner("Searching the docs and writing an answer..."):
        out = run(q.strip())
    if out.get("no_cache"):
        run.clear()
    st.session_state["asked"].add(q)

    with st.container(border=True):
        if out["answer"]:
            st.markdown("#### 💡 Answer")
            st.markdown(out["answer"])
        else:
            st.markdown("#### 🛑 Not covered by the docs")
            st.write(out["refused"])
            st.caption("CiteDocs AI answers only from the FastAPI documentation, "
                       "so it refuses instead of guessing.")

    if out["sources"]:
        st.markdown("#### 📎 Sources" if out["answer"]
                    else "#### 🔎 Closest pages (they did not answer the question)")
        cols = st.columns(len(out["sources"]))
        for col, s in zip(cols, out["sources"]):
            pct = max(0, min(100, int(s["score"] * 100)))
            col.markdown(
                f'<div class="src-card"><span class="src-n">{s["n"]}</span>'
                f'<span class="src-title">{html.escape(s["title"])}</span>'
                f'<div class="bar"><div style="width:{pct}%"></div></div>'
                f'<small>similarity {s["score"]:.2f}</small><br>'
                f'<a href="{html.escape(s["url"])}" target="_blank">Open docs page ↗</a></div>',
                unsafe_allow_html=True)

    st.caption(f"⚡ Retrieval {out['t_ret'] * 1000:.0f} ms · 🤖 LLM {out['t_llm']:.1f} s")

st.divider()
footer = "AI answers can contain mistakes. Always check the linked docs page."
if AUTHOR:
    footer += f" · Built by {html.escape(AUTHOR)}"
st.caption(footer)
