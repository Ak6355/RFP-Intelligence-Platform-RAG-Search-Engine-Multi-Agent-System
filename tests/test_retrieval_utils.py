from search.retrieval_utils import expand_query, make_citation, reciprocal_rank_fusion


def test_query_expansion_keeps_original_and_adds_deadline_terms():
    result = expand_query("What is the submission deadline?")
    assert "submission deadline" in result["expanded_query"]
    assert "due date" in result["expanded_query"]
    assert result["original_query"] == "What is the submission deadline?"


def test_query_expansion_rejects_empty_query():
    try:
        expand_query("   ")
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_rrf_favors_results_present_in_both_lists():
    dense = [
        {"chunk_id": "a", "text": "A"},
        {"chunk_id": "b", "text": "B"},
    ]
    bm25 = [
        {"chunk_id": "b", "text": "B"},
        {"chunk_id": "a", "text": "A"},
    ]
    fused = reciprocal_rank_fusion([dense, bm25])
    assert len(fused) == 2
    assert fused[0]["chunk_id"] == "a" or fused[0]["chunk_id"] == "b"
    assert len(fused[0]["retrieval_methods"]) == 2


def test_rrf_deduplicates_within_same_result_list():
    rows = [{"chunk_id": "a"}, {"chunk_id": "a"}]
    assert len(reciprocal_rank_fusion([rows])) == 1


def test_citation_contains_file_and_page():
    citation = make_citation({"bid_id": "Bid1", "file_name": "Addendum 2.pdf", "page_number": 1})
    assert citation["file"] == "Addendum 2.pdf"
    assert citation["page"] == 1
    assert "page 1" in citation["label"]
