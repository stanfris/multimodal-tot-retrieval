import pytest

from scientific_tot import Dataset, DatasetAdapter
from scientific_tot.data.schemas import DOMAINS


@pytest.mark.live
def test_published_adapter():
    """Opt-in: inventory, small metadata, and one ~7 MB ACL test JSONL; no archives."""
    dataset = Dataset(revision="400ceae164dd6a2414dc1e2c511b06f7942dbeb2")
    inventory = dataset.discover()
    assert {collection.domain for collection in inventory.collections} == set(DOMAINS)
    assert {"metadata", "data", "preprocessed", "query_collections", "splits"} <= set(
        inventory.components
    )
    split = dataset.load_split()
    assert split["train"] and split["test"]
    adapter = DatasetAdapter(dataset)
    queries = adapter.load_queries(domain="ACL", split="test")
    assert queries
    assert all(row["domain"] == "ACL" and row["split"] == "test" for row in queries)
    # Intentionally do not load a corpus: that would download domain TARs.
