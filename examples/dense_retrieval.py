"""Reproduce dense retrieval with pretrained Sentence Transformers and FAISS."""

import argparse
from pathlib import Path

from scientific_tot import Dataset, DatasetAdapter
from scientific_tot.integrations.dense import DEFAULT_MODEL, EncodingConfig, dense_experiment
from scientific_tot.integrations.selection import load_experiment_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument(
        "--revision", help="Dataset commit SHA (local provenance assertion if offline)"
    )
    parser.add_argument("--domain", nargs="+", default=["ACL"])
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision", help="Commit SHA or ref; refs resolve once to a pinned SHA"
    )
    parser.add_argument("--device", default="cuda", help="CUDA embedding device: cuda or cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--encoding-threads", type=int, default=1, help="PyTorch CPU threads")
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
        encoding_threads=args.encoding_threads,
    )
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
    corpus = corpora["fulltext"]
    print(f"Selected {len(corpus)} documents and {len(queries)} queries", flush=True)
    metrics = dense_experiment(
        corpus, queries, args.output, cache=args.cache, config=config, baseline=args.baseline_dir
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
