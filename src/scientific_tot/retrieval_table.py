"""Print a LaTeX table and coverage report from completed benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean

# Allow direct execution with system Python from an uninstalled source checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data.schemas import DOMAINS
else:
    from scientific_tot.data.schemas import DOMAINS

# Model name, representation, display name, modality (in table order).
ROWS = (
    ("bm25", "fulltext", "BM25", "Text"),
    ("dense", "fulltext", "E5", "Text"),
    ("colbert", "fulltext", "ColBERTv2", "Text"),
    ("qwen3_vl_text", "fulltext", "Qwen3-VL", "Text"),
    ("openclip_figures", "figures", "OpenCLIP", "Visual"),
    ("qwen3_vl_figures", "figures", "Qwen3-VL", "Visual"),
    ("openclip", "pages", "OpenCLIP", "Page"),
    ("qwen3_vl_pages", "pages", "Qwen3-VL", "Page"),
    ("qwen3_vl_multimodal", "fulltext", "Qwen3-VL", "Text + Visual"),
)
METRICS = ("RR@10", "R@1", "R@10", "R@100", "index_bytes", "wall_seconds")


def collect_results(root: Path) -> tuple[dict, list[str]]:
    """Keep each dataset's value once; ambiguous repeated runs require selection."""
    if not root.is_dir():
        raise NotADirectoryError(root)
    values, notes = {}, []
    supported = {(model, representation) for model, representation, _, _ in ROWS}
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        stage = manifest.get("stage")
        if stage not in {"evaluate", "index"}:
            continue
        if manifest["status"] != "completed":
            notes.append(f"Skipped {path}: status={manifest['status']}")
            continue
        # Evaluation embeds the retrieval identity, which embeds the index
        # identity. All method/domain metadata survives copying the outputs.
        identity = manifest.get("artifact_identity", {})
        queries = {}
        while "source_identity" in identity:
            identity = identity["source_identity"] or {}
            if "queries" in identity:
                queries = identity["queries"]
        if "index" not in identity:
            notes.append(f"Skipped {path}: missing index identity in manifest")
            continue
        settings = identity["index"]
        model = settings["model"]
        key = (model["name"], settings["representation"]["name"])
        if (
            key not in supported
            or settings.get("test_mode", False)
            or settings["corpus_split"] != "all"
            or (stage == "evaluate" and queries.get("split") != "test")
            or (stage == "evaluate" and queries.get("mode") != "visual-and-text")
            or (model["name"] == "dense" and model["model_id"] != "intfloat/e5-small-v2")
        ):
            notes.append(f"Skipped {path}: outside the full-corpus test benchmark")
            continue
        domains = settings["dataset"]["domains"]
        if len(domains) != 1 or domains[0] not in DOMAINS:
            raise ValueError(f"Expected one benchmark domain: {path}")
        if stage == "evaluate":
            metrics_path = path.parent / "metrics.csv"
            if not metrics_path.is_file():
                notes.append(f"Missing {metrics_path}")
                continue
            with metrics_path.open(newline="") as stream:
                records = list(csv.DictReader(stream))
            if len(records) != 1:
                raise ValueError(f"Expected one aggregate metrics row: {metrics_path}")
            record = records[0]
            metrics = METRICS[:4]
        else:
            record = manifest.get("performance", {})
            metrics = METRICS[4:]
        for metric in metrics:
            raw = record.get(metric)
            if raw is None or raw == "":
                continue
            number = float(raw)
            if not math.isfinite(number) or number < 0 or (stage == "evaluate" and number > 1):
                raise ValueError(f"Invalid {metric}={raw!r}: {path}")
            cell = (*key, metric)
            by_domain = values.setdefault(cell, {})
            domain = domains[0]
            if domain in by_domain:
                raise ValueError(
                    f"Multiple {metric} results for {key}/{domain}: "
                    f"{by_domain[domain][1]} and {path}. "
                    "Select a results root containing one run per method and dataset."
                )
            by_domain[domain] = (number, path)
    return values, notes


def render_table(values: dict) -> str:
    """Macro-average each available cell independently; never impute missing data."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval effectiveness across methods and modalities. "
        r"Each cell is an unweighted mean over available results from the five retrieval "
        r"datasets; missing values are excluded. Index size (decimal GB) and construction "
        r"time (hours, whole index stage) are also averaged over available datasets. "
        r"A dash indicates no available measurements.}",
        r"\label{tab:retrieval_results}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Modality} & \textbf{MRR@10} & \textbf{R@1} & "
        r"\textbf{R@10} & \textbf{R@100} & \textbf{Index Size (GB)} & "
        r"\textbf{Index Time (h)} \\",
        r"\midrule",
    ]
    previous = None
    for model, representation, name, modality in ROWS:
        if previous is not None and previous != modality:
            lines.append(r"\midrule")
        cells = []
        for metric in METRICS:
            samples = values.get((model, representation, metric), {})
            if samples:
                average = mean(number for number, _ in samples.values())
                if metric == "index_bytes":
                    average /= 1e9
                elif metric == "wall_seconds":
                    average /= 3600
                cells.append(format(average, ".3f"))
            else:
                cells.append("-")
        lines.append(" & ".join([name, modality, *cells]) + r" \\")
        previous = modality
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=Path(os.environ.get("TOT_OUTPUT_ROOT", "results")),
        help="root with evaluations/ and indexes/ (default: TOT_OUTPUT_ROOT or results)",
    )
    args = parser.parse_args()
    values, notes = collect_results(args.results)
    print(render_table(values))
    print("Coverage by dataset (each metric has its own denominator):", file=sys.stderr)
    for model, representation, name, modality in ROWS:
        print(f"{name} / {modality}:", file=sys.stderr)
        for metric in METRICS:
            samples = values.get((model, representation, metric), {})
            missing = ", ".join(domain for domain in DOMAINS if domain not in samples) or "none"
            print(f"  {metric}: {len(samples)}/{len(DOMAINS)}; missing: {missing}", file=sys.stderr)
    for note in notes:
        print(note, file=sys.stderr)
    for path in sorted({path for samples in values.values() for _, path in samples.values()}):
        print(f"Source: {path}", file=sys.stderr)
    if not values:
        print("No matching completed benchmark measurements found.", file=sys.stderr)


if __name__ == "__main__":
    main()
