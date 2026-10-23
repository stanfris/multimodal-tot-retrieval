"""Replay one Qwen index batch before committing to a full corpus encoding run."""

import argparse
from contextlib import closing

from scientific_tot.config_arguments import configured_adapter, configured_corpus
from scientific_tot.integrations import visual_dense
from scientific_tot.pbs import is_worker, submit
from scientific_tot.stage_cli import plan


def check_batch(cfg, batch_number):
    import torch

    assert cfg.model.name == "qwen3_vl_multimodal"
    assert batch_number >= 1
    adapter = configured_adapter(cfg)
    corpus = configured_corpus(cfg, adapter)
    rows = visual_dense.document_mapping(corpus, cfg)
    size = cfg.model.batch_size
    total = (len(rows) + size - 1) // size
    if batch_number > total:
        raise ValueError(f"Batch {batch_number} exceeds the corpus's {total} batches")
    selected = rows[(batch_number - 1) * size : batch_number * size]
    print(
        f"Replaying batch {batch_number}/{total}; papers={len(rows)}; batch_size={size}; "
        f"dataset_revision={corpus.provenance['revision']}; model_revision={cfg.model.revision}",
        flush=True,
    )
    encoder = visual_dense.load_model(cfg)
    reserved = [*encoder.model.processor.tokenizer.all_special_tokens, "<|placeholder|>"]
    with closing(
        visual_dense.prefetched_batches(selected, cfg, store=adapter.artifacts)
    ) as batches:
        _, values = next(batches)
        for row in values:
            text = row.get("text", "")
            literals = {token: text.count(token) for token in reserved if token in text}
            print(
                f"Paper {row['doc_id']}: images={len(row['images'])}; "
                f"literal_control_tokens={literals}",
                flush=True,
            )
        with torch.inference_mode():
            embeddings = encoder.encode(values, "document")
        if embeddings.shape != (len(values), cfg.model.dimension):
            raise ValueError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")
        if not torch.isfinite(embeddings).all() or not (embeddings.norm(dim=1) > 0).all():
            raise ValueError("Batch replay returned invalid embeddings")
    print(f"PASS: batch {batch_number} encoded successfully on {cfg.model.device}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "batch", type=int, help="One-based batch number (8912 for the reported failure)"
    )
    parser.add_argument("overrides", nargs="*", help="Same Hydra overrides as the failed index job")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("Batch numbers start at 1")
    cfg, _ = plan("index", args.overrides)
    if cfg.model.name != "qwen3_vl_multimodal":
        parser.error("Batch replay requires experiment=qwen3_vl_multimodal")
    if cfg.launcher.kind == "pbs" and not is_worker():
        submit(
            cfg,
            ["scientific_tot.check_qwen_batch", str(args.batch), *args.overrides],
            name=f"check-qwen-batch-{args.batch}",
        )
        return
    if not cfg.dry_run:
        check_batch(cfg, args.batch)


if __name__ == "__main__":
    main()
