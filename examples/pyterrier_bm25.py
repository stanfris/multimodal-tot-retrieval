"""Reproduce Terrier BM25 on selected scientific ToT documents and queries."""

import argparse
import json
from pathlib import Path

from scientific_tot import Dataset, DatasetAdapter
from scientific_tot.data.schemas import DOMAINS
from scientific_tot.integrations.pyterrier import bm25_experiment
from scientific_tot.integrations.selection import load_experiment_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--domain", choices=DOMAINS, default="ACL")
    parser.add_argument(
        "--corpus-split", choices=["all", "train+test", "train", "test"], default="all"
    )
    parser.add_argument("--query-split", choices=["train", "test", "train+test"], default="test")
    parser.add_argument(
        "--mode", choices=["visual-and-text", "visual-only"], default="visual-and-text"
    )
    parser.add_argument("--missing", choices=["error", "skip"], default="error")
    parser.add_argument(
        "--documents", type=int, default=0, help="Corpus limit; 0 (default) means all documents"
    )
    parser.add_argument(
        "--queries", type=int, default=0, help="Query limit; 0 (default) means all selected queries"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = DatasetAdapter(Dataset(local_dir=args.local_dir, revision=args.revision))
    try:
        corpora, queries = load_experiment_data(
            adapter,
            domain=args.domain,
            corpus_split=args.corpus_split,
            query_split=args.query_split,
            mode=args.mode,
            missing=args.missing,
            documents=args.documents or None,
            query_limit=args.queries or None,
        )
    except ValueError as exc:
        parser.error(
            "\n".join([str(exc), *getattr(exc, "__notes__", [])]).replace(
                "missing='skip'", "--missing skip"
            )
        )
    print(f"Loaded {len(queries)} queries; selected corpora validated", flush=True)
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
