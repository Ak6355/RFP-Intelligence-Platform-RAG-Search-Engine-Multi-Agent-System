PHASE 7 IMPLEMENTATION

This package adds a FastAPI wrapper around the existing Phase 5 search engine and Phase 6 LangGraph CLI graph, plus retrieval evaluation helpers and tests.

COPY/MERGE
- Copy api/ into the project root.
- Copy eval/ into the project root (merge with your existing eval folder; retain your current eval/evaluate.py if needed, but use eval/evaluate_retrieval.py for passage retrieval metrics).
- Copy tests/test_api.py and tests/test_eval_metrics.py into tests/.
- Copy docs/ into docs/.
- Install requirements-phase7.txt into the active .venv.

RUN API
  python -m pip install -r requirements-phase7.txt
  uvicorn api.main:app --host 127.0.0.1 --port 8000
Open http://127.0.0.1:8000/docs

ENDPOINTS
- GET /health
- POST /index  body: {"bid_id":"Bid1"}
- GET /search?q=warranty&bid_id=Bid2&top_k=5
- POST /ask body: {"bid_id":"Bid1","question":"What is the final deadline?"}
- POST /extract body: {"bid_id":"Bid1","max_retries":2}

TEST
  python -m pytest tests/test_api.py tests/test_eval_metrics.py -v
Tests use fake service objects. They do not call Groq, Qdrant or Elasticsearch.

RETRIEVAL EVALUATION
1. Open eval/questions.json.
2. Check every question against the source PDF/HTML and annotate expected_passages with the real file-name substring and page number. For HTML, page_number must be null. Do not leave FILL_PAGE placeholders.
3. The starter set contains 15 questions spanning Bid1 and Bid2. It is a template: verify and correct source hints/pages against your actual files.
4. Run: python -m eval.evaluate_retrieval --questions eval/questions.json --output outputs/retrieval_evaluation.json --top-k 5
5. The script compares dense-only, hybrid RRF, and hybrid RRF + cross-encoder using Recall@5 and MRR.

SAMPLE QA LOG
1. Start the API.
2. Run: python -m eval.run_qa_samples --api http://127.0.0.1:8000
3. This submits 10 sample questions and saves answers/citations to outputs/sample_qa_log.jsonl.

NOTE
The API depends on the exact Phase 5 and Phase 6 packages already copied into search/ and agents/. Verify those phases first. The code is a local development wrapper, not production authentication or public deployment. Protect API keys and do not expose local Qdrant/Elasticsearch ports publicly.
