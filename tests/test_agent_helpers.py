from agents.state import merge_dicts, merge_lists
from agents.extraction_agent import FIELD_GROUPS
from schemas.agent_models import FieldDraft, ExtractionDraft


def test_state_dict_reducer_merges_parallel_agent_outputs():
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_state_list_reducer_appends_trace_events():
    assert merge_lists([{"agent": "retrieval"}], [{"agent": "validator"}]) == [
        {"agent": "retrieval"}, {"agent": "validator"}
    ]


def test_all_twenty_required_fields_are_grouped_once():
    fields = [name for group in FIELD_GROUPS.values() for name in group]
    assert len(fields) == 20
    assert len(set(fields)) == 20
    assert "Due Date" in fields
    assert "Bid Summary" in fields


def test_field_draft_supports_null_not_found_value():
    draft = FieldDraft(value=None, evidence_indexes=[], confidence=0, notes="Not found in documents")
    assert draft.value is None
    assert draft.evidence_indexes == []


def test_extraction_draft_schema_accepts_field_map():
    draft = ExtractionDraft(fields={"Due Date": FieldDraft(value="2030-01-01", evidence_indexes=[0], confidence=0.9)})
    assert draft.fields["Due Date"].value == "2030-01-01"
