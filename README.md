
# RFP Intelligence Platform

AI-powered RFP document intelligence system.

## Features

* PDF and HTML document ingestion
* Document parsing and cleaning
* Chunking and embeddings
* Hybrid RAG search
* Multi-agent orchestration
* Structured bid information extraction
* Addendum reconciliation
* Citation-based question answering

## Project Status

Phase 1: Project initialization.



## PHASE 2 - DOCUMENT INGESTION \& PARSING

## 

## 1\) Copy the 'ingestion' folder from this bundle into the root of your

## &#x20;  RFP\_Intelligence\_Platform project, merging it with the existing folder.

## 

## 2\) Copy tests/test\_parsing.py into your project's tests folder.

## 

## 3\) Activate your environment from PowerShell:

## &#x20;  cd D:\\RFP\_Intelligence\_Platform

## &#x20;  .\\.venv\\Scripts\\Activate.ps1

## 

## 4\) Install Phase 2 libraries:

## &#x20;  python -m pip install "PyMuPDF>=1.24" "pdfplumber>=0.11" "beautifulsoup4>=4.12" "pytest>=8.0"

## 

## 5\) Save dependencies:

## &#x20;  python -m pip freeze > requirements.txt

## 

## 6\) Put the original bid source documents here:

## &#x20;  data\\Bid1\\   (all Bid1 PDFs and HTML files)

## &#x20;  data\\Bid2\\   (all Bid2 PDFs and HTML files)

## 

## 7\) Run ingestion from the project root:

## &#x20;  python -m ingestion.pipeline --data-dir data --output-dir outputs

## 

## 8\) Check these generated files:

## &#x20;  outputs\\normalized\_documents.jsonl

## &#x20;  outputs\\ingestion\_report.json

## &#x20;  logs\\ingestion.log

## 

## 9\) Run the parser tests:

## &#x20;  python -m pytest tests\\test\_parsing.py -v

## 

## NOTES

## \- One JSONL record is created per PDF page. An HTML page creates one record;

## &#x20; its page\_number is null because HTML has no stable page numbering.

## \- The parser extracts PDF text with PyMuPDF and PDF tables with pdfplumber.

## \- Pages with fewer than 50 non-whitespace extracted characters are flagged

## &#x20; as needs\_ocr=true (or empty). OCR itself is NOT performed in Phase 2.

## \- document\_date is populated only when an explicit issue/publication label or

## &#x20; HTML date metadata is found. PDF creation metadata is kept separately.

## \- Unknown file categories are labelled 'other' rather than silently guessed.

## \- This phase does not create chunks, embeddings, vector indexes, BM25 indexes,

## &#x20; or agents. Those are later phases.



RFP INTELLIGENCE PLATFORM - PHASE 3 (CHUNKING)



Purpose

\-------

Convert Phase 2 outputs/normalized\_documents.jsonl into retrieval-ready text

and table chunks. This phase does NOT create embeddings, Qdrant collections,

Elasticsearch indexes, BM25 search, or agents.



Install

\-------

Activate the project's .venv, then run:

&#x20; python -m pip install transformers



Run from the project root

\-------------------------

&#x20; python -m ingestion.chunker --input outputs/normalized\_documents.jsonl --output-dir outputs



Defaults

\--------

Tokenizer: BAAI/bge-large-en-v1.5

Chunk size: 450 tokens

Overlap: 60 tokens



The first run downloads the tokenizer files from Hugging Face. The model's

published sequence length is 512 tokens, so a 450-token content chunk leaves

room for special tokens and small formatting differences. Keep the actual

embedding input within the model's limit in the next phase.



Outputs

\-------

&#x20; outputs/chunks.jsonl

&#x20; outputs/chunking\_report.json



Each chunk keeps source metadata (bid\_id, file\_name, document type, page

number, addendum number, date, and doc\_id where available). PDF page-level

citations remain tied to the original page. Text and extracted tables are

stored as separate content\_type values.



Tests

\-----

Copy tests/test\_chunker.py into the existing tests folder, then run:

&#x20; python -m pytest tests/test\_chunker.py -v



Notes

\-----

1\. Confirm that Phase 2's normalized\_documents.jsonl uses keys such as

&#x20;  text, tables, bid\_id, file\_name, doc\_type, and page\_number. If your Phase 2

&#x20;  schema uses different names, adapt the mapping before running.

2\. Inspect several chunks manually. Automated chunking is not proof that every

&#x20;  specification table or clause is semantically complete.

3\. Do not remove normalized\_documents.jsonl. It is the source for re-running

&#x20;  chunking with different sizes/overlap.







\## Document Chunking Strategy



We use a heading-aware and token-based chunking strategy.



\- Embedding tokenizer: BAAI/bge-large-en-v1.5

\- Chunk size: 450 tokens

\- Overlap: 60 tokens

\- Step size: 390 tokens



\### Why this strategy?



1\. Heading-aware chunking preserves document sections.

2\. Token-based chunking controls the input length.

3\. Overlap reduces information loss at chunk boundaries.

4\. Table-aware chunking preserves specification table headers

&#x20;  and their corresponding rows.

5\. Source metadata is retained for page-level citations.



Text chunks and table chunks are stored separately.



Each chunk contains its original bid ID, filename,

document type and page number wherever available.




RFP INTELLIGENCE PLATFORM - PHASE 3 (CHUNKING)



Purpose

\-------

Convert Phase 2 outputs/normalized\_documents.jsonl into retrieval-ready text

and table chunks. This phase does NOT create embeddings, Qdrant collections,

Elasticsearch indexes, BM25 search, or agents.



Install

\-------

Activate the project's .venv, then run:

&#x20; python -m pip install transformers



Run from the project root

\-------------------------

&#x20; python -m ingestion.chunker --input outputs/normalized\_documents.jsonl --output-dir outputs



Defaults

\--------

Tokenizer: BAAI/bge-large-en-v1.5

Chunk size: 450 tokens

Overlap: 60 tokens



The first run downloads the tokenizer files from Hugging Face. The model's

published sequence length is 512 tokens, so a 450-token content chunk leaves

room for special tokens and small formatting differences. Keep the actual

embedding input within the model's limit in the next phase.



Outputs

\-------

&#x20; outputs/chunks.jsonl

&#x20; outputs/chunking\_report.json



Each chunk keeps source metadata (bid\_id, file\_name, document type, page

number, addendum number, date, and doc\_id where available). PDF page-level

citations remain tied to the original page. Text and extracted tables are

stored as separate content\_type values.



Tests

\-----

Copy tests/test\_chunker.py into the existing tests folder, then run:

&#x20; python -m pytest tests/test\_chunker.py -v



Notes

\-----

1\. Confirm that Phase 2's normalized\_documents.jsonl uses keys such as

&#x20;  text, tables, bid\_id, file\_name, doc\_type, and page\_number. If your Phase 2

&#x20;  schema uses different names, adapt the mapping before running.

2\. Inspect several chunks manually. Automated chunking is not proof that every

&#x20;  specification table or clause is semantically complete.

3\. Do not remove normalized\_documents.jsonl. It is the source for re-running

&#x20;  chunking with different sizes/overlap.







\## Document Chunking Strategy



We use a heading-aware and token-based chunking strategy.



\- Embedding tokenizer: BAAI/bge-large-en-v1.5

\- Chunk size: 450 tokens

\- Overlap: 60 tokens

\- Step size: 390 tokens



\### Why this strategy?



1\. Heading-aware chunking preserves document sections.

2\. Token-based chunking controls the input length.

3\. Overlap reduces information loss at chunk boundaries.

4\. Table-aware chunking preserves specification table headers

&#x20;  and their corresponding rows.

5\. Source metadata is retained for page-level citations.



Text chunks and table chunks are stored separately.



Each chunk contains its original bid ID, filename,

document type and page number wherever available.





RFP INTELLIGENCE PLATFORM - PHASE 4 IMPLEMENTATION

==================================================



WHAT THIS PHASE DOES

\--------------------

1\. Loads the Phase 3 chunks.jsonl file.

2\. Generates dense embeddings with BAAI/bge-large-en-v1.5.

3\. Stores vectors + citation payloads in Qdrant.

4\. Stores chunk text + metadata in Elasticsearch for BM25 keyword search.

5\. Uses stable IDs and upserts so re-running the same chunks does not create duplicates.

6\. Includes a smoke test that searches both indexes independently.



INSTALLATION

\------------

Copy the search/ directory into the root of your existing project, replacing

only the empty search/\_\_init\_\_.py placeholder. Copy tests/test\_indexer.py into

your existing tests/ directory. Copy docker-compose.yml and

requirements-phase4.txt into the project root.



In PowerShell, from D:\\RFP\_Intelligence\_Platform:



&#x20; .\\.venv\\Scripts\\Activate.ps1

&#x20; python -m pip install -r requirements-phase4.txt

&#x20; python -m pip freeze > requirements.txt



START DATABASE SERVICES

\-----------------------

Make sure Docker Desktop is running, then. The compose file binds database ports to 127.0.0.1 so they are not exposed on other network interfaces. Elasticsearch is configured without authentication for local development only.



On Windows with Docker Desktop + WSL2, if Elasticsearch exits during startup with a vm.max\_map\_count error, run this in PowerShell: `wsl -d docker-desktop -u root sysctl -w vm.max\_map\_count=1048576` and restart the service. See Elastic documentation for platform-specific persistence of this setting.



Then:



&#x20; docker compose up -d

&#x20; docker compose ps



Check Qdrant:

&#x20; Invoke-RestMethod http://localhost:6333/collections



Check Elasticsearch:

&#x20; Invoke-RestMethod http://localhost:9200



INDEX PHASE 3 CHUNKS

\--------------------

&#x20; python -m search.indexer --input outputs/chunks.jsonl --output-dir outputs --model BAAI/bge-large-en-v1.5 --batch-size 16



If GPU memory is limited, use --batch-size 4 or --batch-size 8.

The code automatically uses CUDA when PyTorch detects it; otherwise it uses CPU.

The first model load downloads the model from Hugging Face if it is not cached.



TEST BOTH SEARCH INDEXES

\------------------------

&#x20; python -m search.smoke\_test --query "What is the submission deadline?" --bid-id Bid1 --top-k 5



Try a model number query and a warranty query too. Check that the correct

bid/file/page metadata is returned.



RUN UNIT TESTS

\--------------

&#x20; python -m pytest tests\\test\_indexer.py -v



OUTPUT

\------

&#x20; outputs/indexing\_report.json

&#x20; outputs/index\_manifest.json



DATA LOCATIONS

\--------------

Qdrant data: ./storage/qdrant

Elasticsearch data: ./storage/elasticsearch

Both are under the project folder, so they persist between container restarts.



IMPORTANT

\---------

The Elasticsearch configuration disables security for local development only.

Do not expose port 9200 or 6333 to the public internet. Do not use this exact

compose file for production deployment.



The indexing script maintains outputs/index\_manifest.json. On a later run, it

compares stable chunk IDs and content/metadata fingerprints. Unchanged chunks

are skipped, changed chunks are re-embedded and upserted, and newly added bid

chunks are inserted. Use --force-reindex to rebuild all vectors. If an existing

document is replaced and old chunk IDs disappear from chunks.jsonl, stale old

records need deliberate source-level cleanup; this starter implementation does

not automatically delete records that disappear from the source file.



The Phase 3 chunk file must contain non-empty text for each record. Expected

fields include chunk\_id, bid\_id, file\_name, doc\_type, page\_number, text,

content\_type, and chunk\_index. Some optional fields may be absent.





RFP INTELLIGENCE PLATFORM — PHASE 5

===================================



Purpose

\-------

Combine the Phase 4 Qdrant dense-vector index and Elasticsearch BM25 index

into one hybrid search flow with deterministic query expansion, metadata

filters, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and citations.



Files

\-----

search/retrieval\_utils.py  Query expansion, RRF, citation helper (no services)

search/vector\_store.py     Qdrant dense retrieval; supports metadata filters

search/keyword\_search.py   Elasticsearch BM25; supports metadata filters

search/reranker.py         Sentence Transformers CrossEncoder reranker

search/hybrid\_search.py    Orchestrates both searches, RRF, reranking, citations

search/cli.py              Command-line search interface



Install

\-------

1\. Activate the existing project virtual environment.

2\. Copy these files into the matching folders in the project root.

3\. Ensure requirements-phase4.txt exists in the project root.

4\. Run: python -m pip install -r requirements-phase5.txt



Tests (no Qdrant or Elasticsearch connection required)

\------------------------------------------------------

python -m pytest tests/test\_retrieval\_utils.py -v



Search (Qdrant and Elasticsearch must be running and populated from Phase 4)

\---------------------------------------------------------------------------

python -m search.cli --query "What is the final submission deadline?" --bid-id Bid1 --top-k 5



Other examples

\--------------

python -m search.cli --query "Which affidavits are required?" --bid-id Bid2

python -m search.cli --query "What CPU and RAM are required?" --bid-id Bid2 --doc-type specs

python -m search.cli --query "What changed in Addendum 2?" --bid-id Bid1 --addendum-number 2

python -m search.cli --query "Find product details" --bid-id Bid2 --content-type table



If the cross-encoder model cannot be downloaded initially, you can test the

pipeline without reranking using --no-rerank. For the assignment's reranking

requirement, run again without --no-rerank after the model is available.



Design notes

\------------

\- Query expansion keeps the original query and appends conservative

&#x20; procurement-domain synonyms. It does not invent answer content.

\- Both retrieval engines receive the same metadata filters.

\- RRF uses 1 / (rrf\_k + rank), with default rrf\_k=60.

\- The cross-encoder reranks the fused candidate set against the original query.

\- Each result includes source filename, page, bid, document type, addendum

&#x20; number, chunk ID, and a readable citation label.

\- The CLI is the standalone search interface for this phase. A REST API can

&#x20; wrap HybridSearchEngine in a later API phase.



Important compatibility note

\----------------------------

This code assumes the Phase 4 collection and index are both named

"rfp\_chunks" and that Phase 4 stored the text and metadata fields in both

stores. It supports chunk text under the `text` field, as created by the

provided Phase 4 indexer.

\## Hybrid Retrieval Architecture



The platform uses hybrid retrieval combining

dense vector search and BM25 keyword search.



\### Dense Retrieval

Qdrant with BAAI/bge-large-en-v1.5.



\### Keyword Retrieval

Elasticsearch BM25.



\### Query Expansion

Rule-based expansion of common RFP terms,

while preserving the original query.



\### Result Fusion

Reciprocal Rank Fusion (RRF)

with k = 60.



\### Reranking

Cross-Encoder:

cross-encoder/ms-marco-MiniLM-L-6-v2.



\### Metadata Filtering

\- bid\_id

\- doc\_type

\- file\_name

\- addendum\_number

\- content\_type



\### Citations

Every retrieved result retains:

\- Bid ID

\- Original filename

\- Page number

\- Document type

\- Chunk ID

\- Text passage



\### Search Interface

Command-line interface using:



python -m search.cli

RFP INTELLIGENCE PLATFORM — PHASE 6: MULTI-AGENT SYSTEM

=======================================================



What this phase adds

\--------------------

\- LangGraph orchestration with shared typed state.

\- Parallel specialist extraction agents (dates/logistics, commercial/documents,

&#x20; product/company, and bid summary).

\- Retrieval agent that calls the Phase 5 HybridSearchEngine and passes only

&#x20; retrieved passages plus citations to LLM agents (never complete PDFs).

\- Addendum reconciliation agent that applies only explicit changes supported

&#x20; by retrieved addendum evidence.

\- Validator/Critic with deterministic citation checks plus an LLM critic.

\- Bounded re-extraction loop (default maximum: two retries).

\- Q\&A mode with citations tied to retrieved evidence indexes.

\- Ingestion agent that runs the existing Phase 2 parser, Phase 3 chunker, and

&#x20; Phase 4 incremental indexer in sequence.

\- JSON output in outputs/<BidID>.json, QA log in logs/qa\_log.jsonl, and

&#x20; per-extraction agent trace in logs/agent\_trace\_<BidID>.json.



Installation

\------------

1\. Activate the project's virtual environment:

&#x20;  .\\.venv\\Scripts\\Activate.ps1

2\. Copy agents/, schemas/agent\_models.py, tests/test\_agent\_helpers.py,

&#x20;  requirements-phase6.txt and .env.example into the project root.

3\. Install dependencies:

&#x20;  python -m pip install -r requirements-phase6.txt

4\. Copy .env.example to .env and add GROQ\_API\_KEY. Keep .env out of Git.

&#x20;  Default GROQ\_MODEL is openai/gpt-oss-20b; change it to another model

&#x20;  currently available to your Groq account if necessary.

5\. Ensure all previous phase files are installed and Phase 4 databases are

&#x20;  running/populated before extract or ask mode.



Test without calling the LLM

\----------------------------

python -m pytest tests/test\_agent\_helpers.py -v



Extraction mode

\---------------

python -m agents.cli extract --bid-id Bid1

python -m agents.cli extract --bid-id Bid2



Output:

\- outputs/Bid1.json

\- outputs/Bid2.json

\- logs/agent\_trace\_Bid1.json

\- logs/agent\_trace\_Bid2.json



Question answering mode

\------------------------

python -m agents.cli ask --bid-id Bid1 --question "What is the final submission deadline?"

python -m agents.cli ask --bid-id Bid2 --question "Which affidavits are required?"

QA conversations are appended to logs/qa\_log.jsonl.



Ingestion mode

\--------------

python -m agents.cli ingest --bid-id Bid3



The ingestion agent checks data/<BidID>, then runs the existing Phase 2,

Phase 3 and Phase 4 command modules over the current data folder. The Phase 4

manifest is used to skip unchanged chunks while indexing newly added content.



Environment

\-----------

GROQ\_API\_KEY=...

GROQ\_MODEL=openai/gpt-oss-20b

RFP\_TOP\_K\_EVIDENCE=6

RFP\_CANDIDATE\_K=20

RFP\_MAX\_RETRIES=2



Design notes and safety rules

\-----------------------------

\- Agents receive only passages returned by Phase 5 hybrid search.

\- LLM outputs cite zero-based evidence indexes; the application converts those

&#x20; indexes to source citations. Non-null values without a valid evidence index

&#x20; are rejected and retried.

\- Null means not found; do not fill missing fields from outside knowledge.

\- Addendum reconciliation only applies explicit changes with citations.

\- Extraction specialists are dispatched in parallel using LangGraph Send.

\- The validator retries only failed fields, at most max\_retries times.

\- Trace records include timestamp, agent, input summary, search tool calls,

&#x20; output summary, token usage when available, latency and error status.

\- The model-generated confidence is an estimate; it is not a calibrated

&#x20; probability. Human review remains necessary for high-impact bid decisions.



Important integration note

\--------------------------

This starter package expects the Phase 5 HybridSearchEngine API:

&#x20; HybridSearchEngine(enable\_reranker=True)

&#x20; engine.search(query=..., bid\_id=..., top\_k=..., candidate\_k=...)

It expects search results to carry the fields used in Phase 5, including

chunk\_id, text, bid\_id, file\_name, page\_number, doc\_type and citation.

The ingestion mode expects Phase 2 `ingestion.pipeline`, Phase 3

`ingestion.chunker`, and Phase 4 `search.indexer` to be present in the project.

PHASE 7 IMPLEMENTATION



This package adds a FastAPI wrapper around the existing Phase 5 search engine and Phase 6 LangGraph CLI graph, plus retrieval evaluation helpers and tests.



COPY/MERGE

\- Copy api/ into the project root.

\- Copy eval/ into the project root (merge with your existing eval folder; retain your current eval/evaluate.py if needed, but use eval/evaluate\_retrieval.py for passage retrieval metrics).

\- Copy tests/test\_api.py and tests/test\_eval\_metrics.py into tests/.

\- Copy docs/ into docs/.

\- Install requirements-phase7.txt into the active .venv.



RUN API

&#x20; python -m pip install -r requirements-phase7.txt

&#x20; uvicorn api.main:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000/docs



ENDPOINTS

\- GET /health

\- POST /index  body: {"bid\_id":"Bid1"}

\- GET /search?q=warranty\&bid\_id=Bid2\&top\_k=5

\- POST /ask body: {"bid\_id":"Bid1","question":"What is the final deadline?"}

\- POST /extract body: {"bid\_id":"Bid1","max\_retries":2}



TEST

&#x20; python -m pytest tests/test\_api.py tests/test\_eval\_metrics.py -v

Tests use fake service objects. They do not call Groq, Qdrant or Elasticsearch.



RETRIEVAL EVALUATION

1\. Open eval/questions.json.

2\. Check every question against the source PDF/HTML and annotate expected\_passages with the real file-name substring and page number. For HTML, page\_number must be null. Do not leave FILL\_PAGE placeholders.

3\. The starter set contains 15 questions spanning Bid1 and Bid2. It is a template: verify and correct source hints/pages against your actual files.

4\. Run: python -m eval.evaluate\_retrieval --questions eval/questions.json --output outputs/retrieval\_evaluation.json --top-k 5

5\. The script compares dense-only, hybrid RRF, and hybrid RRF + cross-encoder using Recall@5 and MRR.



SAMPLE QA LOG

1\. Start the API.

2\. Run: python -m eval.run\_qa\_samples --api http://127.0.0.1:8000

3\. This submits 10 sample questions and saves answers/citations to outputs/sample\_qa\_log.jsonl.



NOTE

The API depends on the exact Phase 5 and Phase 6 packages already copied into search/ and agents/. Verify those phases first. The code is a local development wrapper, not production authentication or public deployment. Protect API keys and do not expose local Qdrant/Elasticsearch ports publicly.

\# RFP Intelligence Platform



\## Architecture

Explain document ingestion, RAG retrieval,

and multi-agent orchestration.



\## Technology Stack

\- Python

\- PyMuPDF

\- BeautifulSoup

\- Sentence Transformers

\- Qdrant

\- Elasticsearch BM25

\- LangGraph

\- Groq

\- FastAPI



\## Installation

Explain Python environment setup,

dependency installation and Docker setup.



\## Environment Variables

Explain GROQ\_API\_KEY, model names,

Qdrant URL and Elasticsearch URL.



\## Running the Application

Include FastAPI startup command.



\## API Endpoints

Document /index, /search, /ask and /extract.



\## Retrieval Evaluation

Include all 15 questions and actual

Recall@5 and MRR results.



\## Multi-Agent Architecture

Explain each agent and its responsibility.



\## Structured Extraction

Explain all 20 fields, citations,

confidence scores and addendum changes.



\## Testing

Show test commands and actual results.



\## Sample Q\&A

Include at least 10 cited answers.



\## Agent Trace

Include one full extraction trace.



\## Assumptions and Limitations

Document OCR limitations, missing

information handling and known errors.




=======
# RFP-Intelligence-Platform-RAG-Search-Engine-Multi-Agent-System

