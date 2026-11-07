import re

from ingestion.chunker import chunk_record, chunk_text_sections, table_rows_to_chunks


class TinyTokenizer:
    """Deterministic test tokenizer; production runs use the Hugging Face tokenizer."""
    def encode(self, text, add_special_tokens=False):
        return re.findall(r"\S+", text)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_text_chunks_respect_limit_and_overlap():
    tokenizer = TinyTokenizer()
    text = " ".join(f"word{i}" for i in range(35))
    chunks = chunk_text_sections(text, tokenizer, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert all(chunk["token_count"] <= 10 for chunk in chunks)
    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    assert first_words[-2:] == second_words[:2]


def test_heading_is_retained():
    tokenizer = TinyTokenizer()
    chunks = chunk_text_sections("1. DELIVERY REQUIREMENTS\n\nDelivery must be completed within 30 days.", tokenizer, 20, 3)
    assert chunks
    assert chunks[0]["section_heading"] == "1. DELIVERY REQUIREMENTS"
    assert "1. DELIVERY REQUIREMENTS" in chunks[0]["text"]


def test_table_header_is_preserved():
    tokenizer = TinyTokenizer()
    table = [["Product", "CPU"], ["Laptop", "Intel i5"], ["Desktop", "Intel i7"]]
    chunks = table_rows_to_chunks(table, tokenizer, chunk_size=20)
    assert chunks
    assert all("Product | CPU" in chunk["text"] for chunk in chunks)


def test_record_metadata_is_preserved():
    tokenizer = TinyTokenizer()
    record = {
        "doc_id": "doc-1", "bid_id": "Bid1", "file_name": "RFP.pdf",
        "file_type": "pdf", "doc_type": "rfp", "addendum_number": None,
        "page_number": 4, "document_date": None,
        "text": "The bidder shall submit the proposal before the stated deadline.",
        "tables": [],
    }
    chunks = chunk_record(record, tokenizer, chunk_size=12, overlap=2)
    assert chunks
    assert chunks[0]["bid_id"] == "Bid1"
    assert chunks[0]["file_name"] == "RFP.pdf"
    assert chunks[0]["page_number"] == 4
    assert chunks[0]["content_type"] == "text"


def test_table_chunks_keep_content_type_when_in_record():
    tokenizer = TinyTokenizer()
    record = {
        "bid_id": "Bid2", "file_name": "Specs.pdf", "page_number": 2,
        "doc_type": "specs", "text": "", 
        "tables": [[["Product", "RAM"], ["Laptop", "16 GB"]]],
    }
    chunks = chunk_record(record, tokenizer, chunk_size=30, overlap=5)
    assert len(chunks) == 1
    assert chunks[0]["content_type"] == "table"
    assert "Laptop | 16 GB" in chunks[0]["text"]
