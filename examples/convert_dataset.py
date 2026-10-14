"""Convert a selected corpus and test queries to independently usable IR JSONL.

Offline demonstration:
uv run python examples/convert_dataset.py --local-dir tests/fixtures/adapter \
    --domain ACL --output /tmp/scientific-tot-example
"""

import argparse
import json
from pathlib import Path

from scientific_tot import Dataset, DatasetAdapter, validate_qrels
from scientific_tot.data.schemas import DOMAINS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--revision", help="Immutable Hub commit SHA")
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--split", choices=["train", "test", "train+test", "all"], default="all")
    parser.add_argument("--query-split", choices=["train", "test", "train+test"], default="test")
    parser.add_argument("--missing", choices=["error", "skip"], default="error")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    adapter = DatasetAdapter(Dataset(local_dir=args.local_dir, revision=args.revision))
    corpus = adapter.load_corpus(
        domain=args.domain,
        split=args.split,
        representation="fulltext",
        missing=args.missing,
    )
    queries = adapter.load_queries(domain=args.domain, split=args.query_split)
    qrels = validate_qrels(queries, corpus)
    # Validate all three before writing any outputs.
    for name, result in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        result.write_jsonl(args.output / f"{name}.jsonl")
    print(
        json.dumps(
            {
                "documents": len(corpus),
                "queries": len(queries),
                "qrels": len(qrels),
                "coverage": corpus.report.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
