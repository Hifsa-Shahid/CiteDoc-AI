# 📚 CiteDocs AI: cited RAG over the FastAPI documentation

Ask a question in plain English and get an answer with citations, built only from the official FastAPI docs.
If the docs don't cover the question, it says so instead of guessing.

*Unofficial project, not affiliated with FastAPI.*

**Live demo:** <ADD STREAMLIT LINK>

![screenshot](screenshot.png)

## How it works
1. **Chunking:** docs are split by heading (code blocks are never cut) and each chunk gets its page title as context.
2. **Retrieval:** dense embeddings (all-MiniLM-L6-v2) + BM25 keyword search, merged with reciprocal rank fusion.
3. **Reranking (evaluation setup):** a cross-encoder (ms-marco-MiniLM-L-6-v2) re-orders the top candidates.
4. **Generation:** an LLM (Groq, gpt-oss) answers using only the top sources and cites them as [1], [2].
5. **Refusal:** a similarity gate plus an explicit NOT_FOUND check by the model.

## Results
Evaluated on **28 hand-checked questions**: the expected docs page must appear in the results.
With 28 questions, one question is worth about 3.6 percentage points, so small gaps are noise.

| Retrieval method | Top-1 | Top-3 | Top-5 | MRR |
|---|---|---|---|---|
| BM25 (keywords) | 36% | 79% | 86% | 0.57 |
| Dense (embeddings) | 64% | 93% | 96% | 0.78 |
| Hybrid (BM25 + dense) | 68% | 93% | 93% | 0.80 |
| Hybrid + cross-encoder reranker | 86% | 96% | 96% | 0.92 |

**Chunking experiment (dense retrieval):**

| Chunking | Chunks | Top-1 | Top-3 | MRR |
|---|---|---|---|---|
| Split on `#` and `##` | 1000 | 54% | 89% | 0.71 |
| Split on `#`, `##`, `###` | 1573 | 61% | 93% | 0.75 |
| + long sections split | 1645 | 57% | 93% | 0.73 |
| + page title added to each chunk | 1645 | 64% | 93% | 0.78 |

**Refusal behaviour** (similarity gate + LLM "not found" check, measured in the notebook):
- Unrelated questions (cake, weather, ...): 8/8 refused by the similarity gate.
- Near-topic questions about other tools (Flask, Django, Node.js, pandas, ...): similarity gate alone refused only 2/8; with the LLM check, 7/7 genuine off-topic ones were refused.
- Valid FastAPI questions: 28/28 answered, 0 wrongly refused.

**Answer quality** (20 questions graded by hand): <FILL> correct, <FILL> partial, <FILL> wrong.
**Latency:** median retrieval <FILL> ms, median LLM <FILL> s.
**Live demo check** (hybrid retrieval, no reranker): <FILL, e.g. 8/8 valid answered, 7/7 off-topic refused>.

## What I learned
- Keyword search alone was weak (36% top-1) because questions use everyday words while the docs use technical terms.
- The reranker gave the biggest gain: top-1 accuracy rose from 64% (dense) to 86%.
- Adding the page title to each chunk helped (54% to 64% top-1 together with finer splitting); splitting long sections gave no clear benefit.
- A similarity threshold is not enough to refuse questions: Flask and Django questions scored as high as real FastAPI questions. The LLM "not found" check was needed.
- Remaining weak spot: "How do I read headers from the request?" ranked 8th, because many pages mention headers.

## Known limitations
- 28 test questions is a small set, and the refusal thresholds were tuned on the same questions.
- The live app has no reranker (free-tier memory), so its retrieval matches the hybrid row, not the reranker row.
- The free app sleeps after inactivity, and the free LLM tier has rate limits.
- Covers the English FastAPI docs only. AI answers can contain mistakes: always check the linked page.

## Run locally
```
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
streamlit run app.py
```

## Repo contents
- `app.py`: Streamlit app
- `chunks.json`: pre-computed chunks and embeddings
- `results.json`: evaluation numbers shown in the app sidebar
- `notebooks/`: Colab notebook (data prep, evaluation, threshold calibration)
- `evaluation/`: hand-graded answers

## Stack
Python, sentence-transformers, rank_bm25, FAISS, Groq API, Streamlit
