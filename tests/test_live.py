import pytest

from scientific_tot import Dataset


@pytest.mark.live
def test_published_inventory():
    inventory = Dataset(revision="3b764a4c6855925ec802a2560712c4f5cc96f980").discover()
    assert len(inventory.collections) == 10
    assert inventory.components == ("query_collections",)
