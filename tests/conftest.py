import json
import socket
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    if any(
        request.node.get_closest_marker(name)
        for name in ("live", "terrier", "dense_live", "colbert_live", "colqwen_live", "visual_live")
    ):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Default tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def local_dataset(tmp_path):
    folder = tmp_path / "query_collections/ACL/query_generation_test/visual_and_text"
    folder.mkdir(parents=True)
    row = json.loads((FIXTURES / "query.json").read_text())
    (folder / "queries.jsonl").write_text(json.dumps(row) + "\n")
    (folder / "metadata.json").write_text(
        json.dumps(
            {"mode": "visual-and-text", "example_count": 1, "config": {"split_name": "test"}}
        )
    )
    split = tmp_path / "splits"
    split.mkdir()
    (split / "pdf_dataset_split.json").write_text((FIXTURES / "split.json").read_text())
    paper = tmp_path / "preprocessed/ACL/ACL_2022.example-1.1"
    paper.mkdir(parents=True)
    (paper / "paper.json").write_text(
        json.dumps(
            {
                "paper_id": row["metadata"]["paper_id"],
                "source_pdf_relpath": "ACL/2022.example-1.1.pdf",
                "source_paper_dataset": "ACL",
                "markdown_relpath": "markdown.md",
                "figures": [
                    {
                        "figure_id": "authoritative-figure-id",
                        "page_idx": 0,
                        "image_relpath": "images/diagram.png",
                    }
                ],
            }
        )
    )
    (paper / "markdown.md").write_text("# Original scientific text\n")
    return tmp_path
