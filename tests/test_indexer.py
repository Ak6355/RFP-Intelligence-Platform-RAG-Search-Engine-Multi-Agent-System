from search.indexer import stable_chunk_id, _clean_record


def test_stable_id_is_repeatable():
    record = {"chunk_id": "Bid1_file_page1_chunk0"}
    assert stable_chunk_id(record) == stable_chunk_id(record)


def test_stable_id_changes_for_different_chunk_ids():
    first = {"chunk_id": "chunk-a"}
    second = {"chunk_id": "chunk-b"}
    assert stable_chunk_id(first) != stable_chunk_id(second)


def test_clean_record_accepts_text_field():
    record = _clean_record({"chunk_id": "abc", "text": "  Hello RFP  ", "bid_id": "Bid1"})
    assert record["text"] == "Hello RFP"
    assert record["bid_id"] == "Bid1"
    assert record["point_id"]


def test_clean_record_supports_alternate_text_key():
    record = _clean_record({"chunk_id": "abc", "chunk_text": "Hello"})
    assert record["text"] == "Hello"


def test_clean_record_rejects_empty_text():
    try:
        _clean_record({"chunk_id": "abc", "text": "   "})
    except ValueError:
        return
    raise AssertionError("Expected empty text to be rejected")
