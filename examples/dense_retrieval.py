"""Reproduce dense retrieval with pretrained Sentence Transformers and FAISS."""

import argparse
from pathlib import Path

from scientific_tot import Dataset, DatasetAdapter
from scientific_tot.bibliography import title_abstract_from_markdown
from scientific_tot.integrations.dense import DEFAULT_MODEL, EncodingConfig, dense_experiment
from scientific_tot.integrations.pyterrier import select_demo_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument(
        "--revision", help="Dataset commit SHA (local provenance assertion if offline)"
    )
    parser.add_argument("--domain", nargs="+", default=["ACL"])
    parser.add_argument(
        "--corpus-split", choices=["all", "train+test", "train", "test"], default="train+test"
    )
    parser.add_argument("--query-split", choices=["train", "test", "train+test"], default="test")
    parser.add_argument(
        "--mode", choices=["visual-and-text", "visual-only"], default="visual-and-text"
    )
    parser.add_argument(
        "--representation", choices=["fulltext", "title_abstract"], default="fulltext"
    )
    parser.add_argument("--bibliography-from-markdown", action="store_true")
    parser.add_argument("--missing", choices=["error", "skip"], default="error")
    parser.add_argument("--documents", type=int, default=250, help="0 means all selected documents")
    parser.add_argument("--queries", type=int, default=10, help="0 means all selected queries")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision", help="Commit SHA or ref; refs resolve once to a pinned SHA"
    )
    parser.add_argument("--device", default="cpu", help="Embedding device: cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--encoding-profile", choices=["auto", "model", "e5"], default="auto")
    parser.add_argument("--query-prompt")
    parser.add_argument("--document-prompt")
    parser.add_argument("--query-prompt-name")
    parser.add_argument("--document-prompt-name")
    parser.add_argument("--similarity", choices=["cosine", "dot"], default="cosine")
    parser.add_argument("--threads", type=int, default=1, help="FAISS CPU threads")
    parser.add_argument(
        "--baseline-dir", type=Path, help="Require exact identity with a saved BM25 run"
    )
    parser.add_argument("--cache", type=Path, default=Path("results/dense-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = EncodingConfig(
        model_id=args.model,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        profile=args.encoding_profile,
        query_prompt=args.query_prompt,
        document_prompt=args.document_prompt,
        query_prompt_name=args.query_prompt_name,
        document_prompt_name=args.document_prompt_name,
        similarity=args.similarity,
        faiss_threads=args.threads,
    )
    adapter = DatasetAdapter(Dataset(local_dir=args.local_dir, revision=args.revision))
    queries = adapter.load_queries(domain=args.domain, split=args.query_split, mode=args.mode)
    print(f"Loaded {len(queries)} queries; loading canonical corpus...", flush=True)
    # Match the BM25 subset policy before any bibliography projection.
    corpus = adapter.load_corpus(
        domain=args.domain,
        split=args.corpus_split,
        representation="fulltext",
        missing=args.missing,
        include_figures=False,
        include_metadata=False,
    )
    corpus, queries = select_demo_subset(
        corpus, queries, args.documents or None, args.queries or None
    )
    if args.representation == "title_abstract":
        if args.bibliography_from_markdown:
            corpus = title_abstract_from_markdown(corpus, missing=args.missing)
        else:
            wanted = {r["doc_id"] for r in corpus}
            bibliography = adapter.load_corpus(
                domain=args.domain,
                split=args.corpus_split,
                representation="title_abstract",
                missing=args.missing,
                include_figures=False,
            )
            from scientific_tot import Records

            corpus = Records(
                "corpus",
                [r for r in bibliography if r["doc_id"] in wanted],
                bibliography.report,
                bibliography.provenance,
            )
    print(f"Selected {len(corpus)} documents and {len(queries)} queries", flush=True)
    metrics = dense_experiment(
        corpus, queries, args.output, cache=args.cache, config=config, baseline=args.baseline_dir
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
