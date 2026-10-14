"""BM25 retrieval and shared backend-independent saved batch serialization/evaluation."""

import json
import math
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from omegaconf import OmegaConf

from scientific_tot.config_arguments import bm25_arguments, configured_adapter, query_arguments
from scientific_tot.config_artifacts import (
    BATCH_FILES,
    artifact_reference,
    verify_index,
    verify_retrieval,
)
from scientific_tot.integrations.evaluation import evaluate_metrics
from scientific_tot.integrations.pyterrier import _start_terrier, to_pyterrier_queries
from scientific_tot.records import Records, Report


def _indexed_corpus(index, manifest, cfg) -> Records:
    """Use stored docnos and declared domains, never reload or infer corpus content."""
    provenance = manifest["provenance"]
    count = index.getCollectionStatistics().getNumberOfDocuments()
    if count < 1 or count != provenance.get("documents"):
        raise ValueError("Index document count differs from its manifest")
    meta = index.getMetaIndex()
    ids = [str(meta.getItem("docno", i)) for i in range(count)]
    if len(set(ids)) != count:
        raise ValueError("Index contains ambiguous duplicate document IDs")
    domains = provenance.get("document_domains")
    if (
        not isinstance(domains, dict)
        or set(domains) != set(ids)
        or any(domain not in cfg.dataset.domains for domain in domains.values())
    ):
        raise ValueError("Index document identities/domains differ from its manifest")
    rows = [{"doc_id": doc_id, "domain": domains[doc_id]} for doc_id in ids]
    return Records(
        "corpus", rows, Report(selected=count, emitted=count), provenance["corpus_provenance"]
    )


def _validate_batch(run, topics, qrels, mapping, corpus, *, cutoff: int) -> None:
    """Reject incomplete/ambiguous tabular inputs before exporting or computing metrics."""
    for frame, columns in (
        (run, {"qid", "docno", "rank", "score"}),
        (topics, {"qid", "query"}),
        (qrels, {"qid", "docno", "label"}),
        (mapping, {"qid", "query_id", "query_key", "domain"}),
        (corpus, {"doc_id", "domain"}),
    ):
        if not columns <= set(frame.columns) or frame[list(columns)].isna().any().any():
            raise ValueError(f"Incomplete batch columns: {sorted(columns)}")
        for key in columns - {"score", "rank", "label", "query", "query_key"}:
            if (
                not frame[key]
                .map(
                    lambda value: (
                        isinstance(value, str)
                        and bool(value)
                        and not any(c.isspace() for c in value)
                    )
                )
                .all()
            ):
                raise ValueError(f"Invalid canonical identifiers in {key}")
    if topics.empty or corpus.empty or qrels.empty:
        raise ValueError("Batch topics, corpus identities and positive qrels must be nonempty")
    if not topics["query"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Topics must contain nonempty query text")
    if (
        topics.qid.duplicated().any()
        or mapping.qid.duplicated().any()
        or mapping.query_key.duplicated().any()
        or corpus.doc_id.duplicated().any()
        or qrels.duplicated(["qid", "docno"]).any()
        or run.duplicated(["qid", "docno"]).any()
        or run.duplicated(["qid", "rank"]).any()
    ):
        raise ValueError("Ambiguous duplicate batch identifiers/rankings")
    qids, docnos = set(topics.qid), set(corpus.doc_id)
    if (
        set(mapping.qid) != qids
        or set(qrels.qid) != qids
        or not set(run.qid) <= qids
        or not set(qrels.docno) <= docnos
        or not set(run.docno) <= docnos
    ):
        raise ValueError("Batch query/document identities do not resolve to the saved selection")
    unique = mapping.query_id.nunique() == len(mapping)
    expected = (
        mapping.query_id if unique else mapping.query_key.map(lambda key: quote(key, safe=""))
    )
    if not mapping.qid.equals(expected.rename("qid")):
        raise ValueError("Saved query mapping differs from canonical query identifiers")
    doc_domains = dict(zip(corpus.doc_id, corpus.domain, strict=True))
    query_domains = dict(zip(mapping.qid, mapping.domain, strict=True))
    if not (qrels.label == 1).all() or any(
        doc_domains[row.docno] != query_domains[row.qid] for row in qrels.itertuples()
    ):
        raise ValueError("Positive judgments differ from the selected corpus/domain")
    if (
        not run.score.map(
            lambda value: isinstance(value, (int, float)) and math.isfinite(value)
        ).all()
        or not run["rank"].map(lambda value: isinstance(value, int) and 1 <= value <= cutoff).all()
        or (run.groupby("qid").size() > cutoff).any()
    ):
        raise ValueError("Invalid ranking score/rank or results exceed the saved retrieval cutoff")
    for _, group in run.groupby("qid", sort=False):
        ordered = group.sort_values("rank")
        if ordered["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError("Ranking positions are incomplete or ambiguous")
        if not ordered.score.is_monotonic_decreasing:
            raise ValueError("Ranking positions disagree with scores")


def _seal_outputs(target: Path, names) -> dict:
    references = {name: artifact_reference(target / name) for name in names}
    for name in names:
        (target / name).chmod(0o444)
    return references


def retrieve_bm25(cfg, target: Path) -> dict:
    """Open a verified persistent index and retrieve independently selected queries."""
    location = Path(cfg.paths.index)
    print(f"Verifying Terrier index: {location}", flush=True)
    manifest = verify_index(location, cfg)
    properties = location / "data.properties"
    if not properties.is_file():
        raise FileNotFoundError(f"Incomplete Terrier index: {properties}")
    if manifest["provenance"].get("terrier_version") != cfg.launcher.terrier_version:
        raise ValueError("Requested Terrier version differs from the saved index")
    adapter = configured_adapter(cfg)
    queries = adapter.load_queries(**query_arguments(cfg))
    query_references = [
        {"dataset_path": name, **artifact_reference(adapter.dataset.get_file(name))}
        for name in queries.provenance["selection"]["files"]
    ]
    import pyterrier as pt

    _start_terrier(pt, cfg.launcher.terrier_version)
    index = pt.terrier.IndexFactory.of(str(properties))
    if index is None:
        raise ValueError(f"Cannot open persistent Terrier index: {properties}")
    try:
        corpus = _indexed_corpus(index, manifest, cfg)
        inputs = to_pyterrier_queries(corpus, queries)
        print(
            f"BM25 retrieval: {len(corpus):,} indexed documents, {len(inputs.topics):,} queries; "
            f"query_split={cfg.split.queries}, top_k={cfg.retrieval.top_k}",
            flush=True,
        )
        _, arguments = bm25_arguments(
            cfg, docno_bytes=max(len(row["doc_id"].encode("utf-8")) for row in corpus)
        )
        run = pt.terrier.Retriever(index, verbose=True, **arguments).transform(inputs.topics)
    finally:
        index.close()
    # PyTerrier uses zero-based ranks internally; the exported TREC ranks start at one.
    run = run[["qid", "docno", "rank", "score"]].copy()
    run["rank"] += 1
    mapping = inputs.query_ids.copy()
    mapping["domain"] = mapping.query_key.map({row["query_key"]: row["domain"] for row in queries})
    identities = pd.DataFrame(list(corpus), columns=["doc_id", "domain"])
    outputs = save_batch(run, inputs, mapping, identities, cfg, target)
    return {
        "inputs": {
            "index_manifest": artifact_reference(location / "index-manifest.json"),
            "index_properties": artifact_reference(properties),
            "query_files": query_references,
        },
        "outputs": outputs,
        "queries": len(inputs.topics),
        "qrels": len(inputs.qrels),
        "rankings": len(run),
        "query_provenance": queries.provenance,
        "corpus_sha256": manifest["provenance"]["corpus_sha256"],
        "dataset_revision": manifest["provenance"]["dataset_revision"],
        "retrieval_arguments": arguments,
        "versions": {name: version(name) for name in ("pyterrier", "pandas")},
    }


def save_batch(run, inputs, mapping, identities, cfg, target: Path) -> dict:
    """Shared paper-level batch contract; standard serialization belongs to PyTerrier/pandas."""
    import pyterrier as pt

    print(f"Saving {len(run):,} rankings: {target}", flush=True)
    _validate_batch(
        run, inputs.topics, inputs.qrels, mapping, identities, cutoff=cfg.retrieval.top_k
    )
    run = run.sort_values(["qid", "rank"], kind="stable")
    pt.io.write_results(run, str(target / "run.trec"), format="trec", run_name=cfg.run.name)
    inputs.topics.to_json(target / "topics.jsonl", orient="records", lines=True, force_ascii=False)
    inputs.qrels.to_csv(target / "qrels.tsv", sep="\t", index=False)
    mapping.to_json(target / "query-ids.jsonl", orient="records", lines=True, force_ascii=False)
    identities.to_json(target / "corpus-ids.jsonl", orient="records", lines=True, force_ascii=False)
    return _seal_outputs(target, BATCH_FILES)


def _read_jsonl(path: Path) -> pd.DataFrame:
    # Construct directly from JSON so numeric-looking canonical identifiers stay strings.
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Incomplete JSONL artifact: {path}")
    return pd.DataFrame(rows)


def evaluate_saved_run(cfg, target: Path) -> dict:
    """Compute YAML metrics from a completed saved batch, without dataset/index reads."""
    source = Path(cfg.paths.run)
    print(f"Loading and verifying saved retrieval run: {source}", flush=True)
    manifest = verify_retrieval(source, cfg)
    built = OmegaConf.load(source / ".hydra/config.yaml")
    topics = _read_jsonl(source / "topics.jsonl")
    mapping = _read_jsonl(source / "query-ids.jsonl")
    corpus = _read_jsonl(source / "corpus-ids.jsonl")
    qrels = pd.read_csv(
        source / "qrels.tsv",
        sep="\t",
        dtype={"qid": str, "docno": str, "label": int},
        keep_default_na=False,
    )
    # Read the standard six-column table with NA conversion disabled for identifiers such as "NA".
    if (source / "run.trec").stat().st_size == 0:
        run = pd.DataFrame(columns=["qid", "docno", "rank", "score", "name"])
    else:
        run = pd.read_csv(
            source / "run.trec", sep=r"\s+", header=None, dtype=str, keep_default_na=False
        )
        if run.shape[1] != 6 or run.isna().any().any():
            raise ValueError("Incomplete TREC rankings: expected six columns")
        run.columns = ["qid", "iter", "docno", "rank", "score", "name"]
        if set(run["name"]) != {built.run.name} or set(run["iter"]) != {"Q0"}:
            raise ValueError("Ambiguous TREC input: expected the saved experiment's single run")
        run["rank"] = pd.to_numeric(run["rank"], errors="raise")
        run["score"] = pd.to_numeric(run["score"], errors="raise")
    _validate_batch(run, topics, qrels, mapping, corpus, cutoff=built.retrieval.top_k)
    metrics, per_query = evaluate_metrics(
        run, topics, qrels, name=built.run.name, evaluation=cfg.evaluation
    )
    per_query = per_query.merge(mapping, on="qid", how="left", validate="many_to_one")
    metrics.to_csv(target / "metrics.csv", index=False)
    per_query.to_csv(target / "per-query.csv", index=False)
    return {
        "inputs": {
            "retrieval_manifest": artifact_reference(source / "manifest.json"),
            "retrieval_configuration": artifact_reference(source / ".hydra/config.yaml"),
            "batch": manifest["artifacts"]["outputs"],
        },
        "outputs": _seal_outputs(target, ("metrics.csv", "per-query.csv")),
        "metrics": list(cfg.evaluation.metrics),
        "queries": len(topics),
        "retrieval_cutoff": built.retrieval.top_k,
        "aggregation": manifest["artifacts"].get("aggregation"),
        "ranking_level": "paper",
        "versions": {name: version(name) for name in ("pyterrier", "ir-measures", "pandas")},
    }
