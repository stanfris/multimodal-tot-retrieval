"""Reproduce Terrier BM25 on selected scientific ToT documents and queries."""

import argparse
import json
from pathlib import Path

from scientific_tot import Dataset, DatasetAdapter, Records
from scientific_tot.bibliography import title_abstract_from_markdown
from scientific_tot.integrations.pyterrier import bm25_experiment, select_demo_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--domain", nargs="+", default=["ACL"])
    parser.add_argument(
        "--corpus-split", choices=["all", "train+test", "train", "test"], default="train+test"
    )
    parser.add_argument("--query-split", choices=["train", "test", "train+test"], default="test")
    parser.add_argument(
        "--mode", choices=["visual-and-text", "visual-only"], default="visual-and-text"
    )
    parser.add_argument(
        "--representation", choices=["fulltext", "title_abstract", "both"], default="both"
    )
    parser.add_argument(
        "--bibliography-from-markdown",
        action="store_true",
        help="Explicitly select original title/Abstract Markdown sections",
    )
    parser.add_argument("--missing", choices=["error", "skip"], default="error")
    parser.add_argument(
        "--documents", type=int, help="Demo corpus size; positive targets always included"
    )
    parser.add_argument("--queries", type=int, help="Number of queries in the reproducible demo")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = DatasetAdapter(Dataset(local_dir=args.local_dir, revision=args.revision))
    queries = adapter.load_queries(domain=args.domain, split=args.query_split, mode=args.mode)
    print(f"Loaded {len(queries)} queries; loading selected corpus...", flush=True)
    fulltext = adapter.load_corpus(
        domain=args.domain,
        split=args.corpus_split,
        representation="fulltext",
        missing=args.missing,
        include_figures=False,
        include_metadata=False,
    )
    print(
        f"Loaded {len(fulltext)} usable documents; "
        f"skipped {fulltext.report.skipped}, issues {len(fulltext.report.issues)}",
        flush=True,
    )
    fulltext, queries = select_demo_subset(fulltext, queries, args.documents, args.queries)
    corpora = {}
    if args.representation in {"fulltext", "both"}:
        corpora["fulltext"] = fulltext
    if args.representation in {"title_abstract", "both"}:
        if args.bibliography_from_markdown:
            corpora["title_abstract"] = title_abstract_from_markdown(fulltext, missing=args.missing)
        else:
            bibliography = adapter.load_corpus(
                domain=args.domain,
                split=args.corpus_split,
                representation="title_abstract",
                missing=args.missing,
                include_figures=False,
            )
            wanted = {r["doc_id"] for r in fulltext}
            rows = [r for r in bibliography if r["doc_id"] in wanted]
            corpora["title_abstract"] = Records(
                "corpus", rows, bibliography.report, bibliography.provenance
            )
    if len(corpora) == 2 and {r["doc_id"] for r in corpora["fulltext"]} != {
        r["doc_id"] for r in corpora["title_abstract"]
    }:
        raise ValueError(
            "Representations have different corpus IDs; choose a covered subset explicitly"
        )
    # Conversion validates all qrels before either representation is indexed.
    from scientific_tot.integrations.pyterrier import to_pyterrier

    for corpus in corpora.values():
        to_pyterrier(corpus, queries)
    for representation, corpus in corpora.items():
        print(representation, flush=True)
        metrics = bm25_experiment(
            corpus, queries, args.output / representation, threads=args.threads
        )
        print(metrics.to_string(index=False), flush=True)
        (args.output / representation / "selected-documents.json").write_text(
            json.dumps([r["doc_id"] for r in corpus], indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
