import json

import pytest

pytest.importorskip("pandas")

from scientific_tot import Records
from scientific_tot.bibliography import title_abstract_from_markdown
from scientific_tot.integrations.pyterrier import to_pyterrier
from scientific_tot.records import IncompleteCorpusError, Report, UnresolvedRelevanceError


def corpus(rows):
    return Records("corpus", rows, Report(selected=len(rows), emitted=len(rows)), {})


def queries(rows):
    return Records("queries", rows, Report(selected=len(rows), emitted=len(rows)), {})


def document(doc_id="ACL_one", text="zebra scientific research"):
    return {
        "doc_id": doc_id,
        "text": text,
        "title": None,
        "abstract": None,
        "domain": "ACL",
        "representation": "fulltext",
        "provenance": {"content": {"path": "markdown.md"}},
    }


def query(query_id="q1", query_key="scope-q1", text="zebra scientific"):
    return {
        "query_id": query_id,
        "query_key": query_key,
        "text": text,
        "domain": "ACL",
        "split": "test",
        "mode": "visual-and-text",
        "relevant_ids": ["ACL_one"],
    }


def test_canonical_conversion_preserves_docnos_and_query_ids():
    result = to_pyterrier(corpus([document()]), queries([query()]))
    assert result.documents == ({"docno": "ACL_one", "text": "zebra scientific research"},)
    assert result.topics.to_dict("records") == [{"qid": "q1", "query": "zebra scientific"}]
    assert result.qrels.to_dict("records") == [{"qid": "q1", "docno": "ACL_one", "label": 1}]


def test_colliding_query_ids_use_reversible_whitespace_free_mapping():
    key1 = json.dumps(["ACL", "test", "visual_and_text", "q1"])
    key2 = json.dumps(["ACL", "test", "visual_only", "q1"])
    result = to_pyterrier(
        corpus([document()]), queries([query(query_key=key1), query(query_key=key2)])
    )
    assert result.topics.qid.nunique() == 2
    assert set(result.qrels.qid) == set(result.topics.qid)
    assert not any(c.isspace() for qid in result.topics.qid for c in qid)
    from urllib.parse import unquote

    assert {unquote(qid) for qid in result.topics.qid} == {key1, key2}
    assert result.query_ids.query_id.tolist() == ["q1", "q1"]


def test_unresolved_qrels_prevent_framework_conversion():
    with pytest.raises(UnresolvedRelevanceError):
        to_pyterrier(corpus([document("ACL_other")]), queries([query()]))


def test_unsafe_trec_docno_rejected():
    q = query()
    q["relevant_ids"] = ["ACL with spaces"]
    with pytest.raises(ValueError, match="whitespace-free"):
        to_pyterrier(corpus([document("ACL with spaces")]), queries([q]))


def test_literal_bibliography_selects_original_spans():
    source = (
        "# Original title\n\nAuthors\n\n## Abstract\n\nOriginal abstract.\n\n"
        "## 1 Introduction\nBody."
    )
    result = title_abstract_from_markdown(corpus([document(text=source)]))
    assert result[0]["title"] == "Original title"
    assert result[0]["abstract"] == "Original abstract."
    assert result[0]["text"] == "Original title\n\nOriginal abstract."
    spans = result[0]["provenance"]["bibliography_projection"]["character_spans"]
    assert source[slice(*spans["title"])] == result[0]["title"]
    assert source[slice(*spans["abstract"])].strip() == result[0]["abstract"]
    assert result[0]["doc_id"] == "ACL_one"


def test_unrecognized_abstract_is_not_fabricated():
    result = title_abstract_from_markdown(corpus([document(text="# Title\nUnknown prose")]))
    assert result[0]["abstract"] is None
    assert result.report.issues[0].code == "partial_title_abstract"
    with pytest.raises(IncompleteCorpusError):
        title_abstract_from_markdown(corpus([document(text="No identifiable headings")]))


def test_explicit_bibliography_metadata_wins():
    row = document(text="# Other heading\n## Abstract\nText\n## Body\nBody")
    row.update(title="Authoritative title", abstract="Authoritative abstract")
    result = title_abstract_from_markdown(corpus([row]))
    assert result[0]["text"] == "Authoritative title\n\nAuthoritative abstract"


def test_demo_subset_retains_positives_and_is_deterministic():
    from scientific_tot.integrations.pyterrier import select_demo_subset

    docs = corpus([document("ACL_z"), document("ACL_one"), document("ACL_a")])
    selected, topics = select_demo_subset(docs, queries([query()]), 2, 1)
    assert {r["doc_id"] for r in selected} == {"ACL_one", "ACL_a"}
    assert len(to_pyterrier(selected, topics).qrels) == 1


@pytest.mark.terrier
def test_standard_terrier_index_run_and_metrics(tmp_path):
    from scientific_tot.integrations.pyterrier import bm25_experiment

    docs = corpus(
        [
            document(),
            document("ACL_two", "whale ocean biology"),
            document("ACL_" + "long" * 20, "computer dialogue language"),
        ]
    )
    qs = queries([query(), query("q2", "scope-q2", "unfindablexyz")])
    metrics = bm25_experiment(docs, qs, tmp_path / "experiment")
    assert metrics.iloc[0]["RR@10"] == pytest.approx(0.5)
    assert metrics.iloc[0]["R@10"] == pytest.approx(0.5)
    assert metrics.iloc[0]["R@1"] == pytest.approx(0.5)
    assert metrics.iloc[0]["nDCG@10"] == pytest.approx(0.5)
    import pyterrier as pt

    run = pt.io.read_results(str(tmp_path / "experiment/run.trec"))
    assert set(run.docno).issubset({d["doc_id"] for d in docs})
    assert run.iloc[0].docno == "ACL_one"
    manifest = json.loads((tmp_path / "experiment/manifest.json").read_text())
    assert manifest["doc_ids_verified_against_index"] is True
    assert manifest["queries"] == 2
    with pytest.raises(FileExistsError):
        bm25_experiment(docs, qs, tmp_path / "experiment")


def test_independently_loaded_qrels_are_validated():
    from scientific_tot.adapter import validate_qrels

    docs, topics = corpus([document()]), queries([query()])
    qrels = validate_qrels(topics, docs)
    assert to_pyterrier(docs, topics, qrels).qrels.iloc[0].docno == "ACL_one"
    with pytest.raises(ValueError, match="Supplied canonical qrels"):
        to_pyterrier(docs, topics, Records("qrels", [], Report(), {}))


def test_hydra_settings_reach_external_terrier_arguments(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd
    from hydra import compose, initialize_config_dir

    from scientific_tot.config_arguments import (
        configured_adapter,
        corpus_arguments,
        query_arguments,
    )
    from scientific_tot.configuration import CONFIG_DIR, resolve_config
    from scientific_tot.integrations.pyterrier import bm25_experiment

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset.root={Path(__file__).parent / 'fixtures/adapter'}",
                "model.indexing.blocks=true",
                "model.indexing.meta.docno=256",
                "model.retrieval.controls={bm25.b:0.5,bm25.k_1:2.0}",
                "retrieval.top_k=7",
                "launcher.threads.indexing=3",
                "evaluation.metrics=[RR@5]",
            ],
        )
    cfg = resolve_config(cfg, tmp_path)
    adapter = configured_adapter(cfg)
    docs, topics = (
        adapter.load_corpus(**corpus_arguments(cfg)),
        adapter.load_queries(**query_arguments(cfg)),
    )
    calls = []
    ids = [r["doc_id"] for r in docs]
    index = SimpleNamespace(
        getCollectionStatistics=lambda: SimpleNamespace(getNumberOfDocuments=lambda: len(ids)),
        getMetaIndex=lambda: SimpleNamespace(getItem=lambda key, number: ids[number]),
        close=lambda: None,
    )
    classic = object()

    class Indexer:
        def __init__(self, path, **kwargs):
            calls.append(("indexer", path, kwargs))

        def index(self, rows):
            assert len(list(rows)) == 2
            return index

    class Retriever:
        def __init__(self, location, **kwargs):
            assert location is index
            calls.append(("retriever", kwargs))

        def transform(self, topics):
            return pd.DataFrame()

    monkeypatch.setitem(
        sys.modules,
        "pyterrier",
        SimpleNamespace(
            tqdm=lambda rows, **kwargs: rows,
            java=SimpleNamespace(started=lambda: True),
            terrier=SimpleNamespace(
                IterDictIndexer=Indexer,
                Retriever=Retriever,
                IndexingType=SimpleNamespace(CLASSIC=classic),
                IndexFactory=SimpleNamespace(of=lambda ref: ref),
                version=lambda: "5.11",
            ),
        ),
    )

    def evaluate(*args, **kwargs):
        calls.append(("evaluation", list(kwargs["evaluation"].metrics)))
        return pd.DataFrame()

    monkeypatch.setattr("scientific_tot.integrations.evaluation.evaluate_run", evaluate)
    bm25_experiment(docs, topics, tmp_path / "run", settings=cfg)
    assert calls[0][2] == {
        "type": classic,
        "blocks": True,
        "meta": {"docno": 256},
        "text_attrs": ["text"],
        "threads": 3,
    }
    assert calls[1][1] == {
        "wmodel": "BM25",
        "controls": {"bm25.b": 0.5, "bm25.k_1": 2.0},
        "num_results": 7,
        "threads": 3,
        "verbose": True,
    }
    assert calls[2] == ("evaluation", ["RR@5"])
