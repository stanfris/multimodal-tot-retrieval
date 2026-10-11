import pytest

from scientific_tot import Dataset, DatasetAdapter
from scientific_tot.data.hub import MissingComponentError


@pytest.mark.live
def test_published_adapter():
    """Opt-in: inventory, small metadata, and one ~7 MB ACL test JSONL; no archives."""
    dataset = Dataset(revision="3b764a4c6855925ec802a2560712c4f5cc96f980")
    inventory = dataset.discover()
    assert len(inventory.collections) == 10
    assert inventory.components == ("query_collections",)
    adapter = DatasetAdapter(dataset)
    queries = adapter.load_queries(domain="ACL", split="test")
    assert len(queries) == 110
    assert queries[0]["query_id"] == "test_visual_and_text_q00000"
    assert queries[0]["relevant_ids"] == ["ACL_2022.coling-1.30"]
    with pytest.raises(MissingComponentError):
        adapter.load_corpus(domain="ACL")
