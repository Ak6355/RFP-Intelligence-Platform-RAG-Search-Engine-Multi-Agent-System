# 5–10 minute demo script

1. Show the project structure and explain the pipeline: ingestion -> chunks -> indexes -> hybrid retrieval -> LangGraph -> FastAPI.
2. Open `/docs` and show the four endpoints: `/index`, `/search`, `/ask`, `/extract`.
3. Run a Bid1 search and point out filename/page citations.
4. Ask a Bid1 question about the final due date and show the cited answer.
5. Run Bid1 extraction and open `outputs/Bid1.json`; show field values, nulls, citations, confidence and validation summary.
6. Repeat a Bid2 question about affidavits or specifications.
7. Open the agent trace and show the retrieval, extraction, addendum and validation events.
8. Show the retrieval evaluation report and compare dense-only, hybrid RRF, and hybrid+reranker metrics.
9. State any known limitations honestly, including OCR coverage, any unannotated evaluation passages, and any failures on scanned pages.
