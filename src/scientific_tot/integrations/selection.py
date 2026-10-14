"""Full-document corpus/query selection shared by retrieval examples."""

from scientific_tot import DatasetAdapter, Records
from scientific_tot.adapter import validate_qrels
from scientific_tot.integrations.pyterrier import select_demo_subset


def load_experiment_data(
    adapter: DatasetAdapter,
    *,
    domain: str = "ACL",
    corpus_split: str = "all",
    query_split: str = "test",
    mode: str = "visual-and-text",
    missing: str = "error",
    documents: int | None = None,
    query_limit: int | None = None,
) -> tuple[dict[str, Records], Records]:
    """Load fulltext and validate all positives before indexing or encoding.

    All domain documents and all selected test queries are used by default.
    Explicit positive limits select a deterministic demo including every positive.
    Bibliographic fields are not required or loaded from the PDF metadata index.
    """
    queries = adapter.load_queries(domain=domain, split=query_split, mode=mode)
    corpus = adapter.load_corpus(
        domain=domain,
        split=corpus_split,
        representation="fulltext",
        missing=missing,
        include_figures=False,
        include_metadata=False,
    )
    corpus, queries = select_demo_subset(corpus, queries, documents, query_limit)
    if not corpus or not queries:
        raise ValueError("Experiment needs a nonempty corpus and query set")
    validate_qrels(queries, corpus)
    return {"fulltext": corpus}, queries
