"""Small tests for the Phase 2 parsers and cleaning helpers."""

from pathlib import Path

import fitz

from ingestion.cleaner import clean_text, normalize_table
from ingestion.file_detector import detect_file_type, extract_addendum_number, infer_doc_type
from ingestion.html_parser import parse_html
from ingestion.pdf_parser import parse_pdf


def test_clean_text_joins_broken_hyphenated_word_and_wrapped_line():
    raw = "The require-\nments must be met\nby the bidder."
    cleaned = clean_text(raw)
    assert "requirements" in cleaned
    assert "must be met by the bidder." in cleaned


def test_table_normalization_removes_empty_rows():
    assert normalize_table([["Name", "Value"], [None, "  "], ["CPU", "Intel"]]) == [
        ["Name", "Value"],
        ["CPU", "Intel"],
    ]


def test_file_detection_and_metadata():
    pdf_path = Path("Addendum 2.pdf")
    assert detect_file_type(pdf_path) == "pdf"
    assert infer_doc_type(pdf_path, "pdf") == "addendum"
    assert extract_addendum_number(pdf_path) == 2
    assert infer_doc_type(Path("Dell Laptop Specs.pdf"), "pdf") == "specs"
    assert infer_doc_type(Path("BidNet page.html"), "html") == "bid_page"


def test_html_parser_extracts_title_table_and_text(tmp_path):
    html_path = tmp_path / "bid.html"
    html_path.write_text(
        """<!doctype html><html><head><title>Test Bid</title>
        <meta name='datePublished' content='2026-10-01'></head><body>
        <main><h1>Computer Bid</h1><p>Submit your response online.</p>
        <table><tr><th>Item</th><th>Quantity</th></tr>
        <tr><td>Laptop</td><td>20</td></tr></table></main>
        <script>ignore this</script></body></html>""",
        encoding="utf-8",
    )
    result = parse_html(html_path)
    assert result["title"] == "Test Bid"
    assert "Submit your response online" in result["text"]
    assert result["tables"][0][1] == ["Laptop", "20"]
    assert result["document_date"] == "2026-10-01"


def test_pdf_parser_extracts_page_text_and_page_number(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "This is a sample RFP page with enough text to test extraction.")
    document.save(pdf_path)
    document.close()

    result = parse_pdf(pdf_path)
    assert len(result["pages"]) == 1
    assert result["pages"][0]["page_number"] == 1
    assert "sample RFP" in result["pages"][0]["raw_text"]
