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


def test_independently_loaded_qrels_are_validated():
    from scientific_tot.adapter import validate_qrels

    docs, topics = corpus([document()]), queries([query()])
    qrels = validate_qrels(topics, docs)
    assert to_pyterrier(docs, topics, qrels).qrels.iloc[0].docno == "ACL_one"
    with pytest.raises(ValueError, match="Supplied canonical qrels"):
        to_pyterrier(docs, topics, Records("qrels", [], Report(), {}))
