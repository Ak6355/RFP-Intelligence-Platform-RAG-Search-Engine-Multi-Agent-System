from fastapi.testclient import TestClient
from api.main import create_app


class FakeServices:
    project_root = "."
    def search(self, **kwargs):
        return {"query": kwargs["query"], "results": [{"file_name": "sample.pdf", "page_number": 2}]}
    def invoke_agent(self, mode, bid_id, *, question=None, max_retries=2):
        if mode == "qa":
            return {"answer": "Sample grounded answer", "answer_sources": [{"file": "sample.pdf", "page": 2}], "confidence": 0.9}
        if mode == "extract":
            return {"final_json_path": None, "validation_report": {"passed": 20}}
        return {"ingestion_result": {"status": "completed", "indexed": 4}}


def test_health():
    client = TestClient(create_app(services=FakeServices()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search():
    client = TestClient(create_app(services=FakeServices()))
    response = client.get("/search", params={"q": "warranty", "bid_id": "Bid2"})
    assert response.status_code == 200
    assert response.json()["results"][0]["page_number"] == 2


def test_ask():
    client = TestClient(create_app(services=FakeServices()))
    response = client.post("/ask", json={"bid_id": "Bid1", "question": "What is the deadline?"})
    assert response.status_code == 200
    assert response.json()["citations"]


def test_extract():
    client = TestClient(create_app(services=FakeServices()))
    response = client.post("/extract", json={"bid_id": "Bid1"})
    assert response.status_code == 200
    assert response.json()["validation"]["passed"] == 20


def test_invalid_bid_id_rejected():
    client = TestClient(create_app(services=FakeServices()))
    response = client.post("/ask", json={"bid_id": "../secret", "question": "What is this?"})
    assert response.status_code == 422
