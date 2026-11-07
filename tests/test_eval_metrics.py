import pytest
from eval.metrics import annotation_complete, recall_at_k, reciprocal_rank


def test_recall_at_k():
    results = [{"file_name": "Addendum 2.pdf", "page_number": 1}, {"file_name": "RFP.pdf", "page_number": 3}]
    gold = [{"file_contains": "Addendum 2", "page_number": 1}]
    assert recall_at_k(results, gold, 5) == 1.0


def test_mrr_first_relevant():
    results = [{"file_name": "RFP.pdf", "page_number": 2}, {"file_name": "Addendum 2.pdf", "page_number": 1}]
    gold = [{"file_contains": "Addendum 2", "page_number": 1}]
    assert reciprocal_rank(results, gold) == 0.5


def test_missing_annotation_rejected():
    with pytest.raises(ValueError):
        recall_at_k([], [{"file_contains": "Addendum 2", "page_number": "FILL_PAGE"}], 5)


def test_annotation_complete_html():
    assert annotation_complete({"expected_passages": [{"file_contains": "BidNet", "page_number": None}]})


def test_annotation_incomplete_pdf():
    assert not annotation_complete({"expected_passages": [{"file_contains": "RFP", "page_number": "FILL_PAGE"}]})
