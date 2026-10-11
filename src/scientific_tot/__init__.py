"""Scientific ToT dataset adapters; no retrieval or generation runtime."""

from scientific_tot.adapter import DatasetAdapter, validate_qrels
from scientific_tot.api import load_corpus, load_metadata, load_qrels, load_queries
from scientific_tot.data.hub import Dataset
from scientific_tot.records import (
    IncompleteCorpusError,
    Records,
    UnresolvedRelevanceError,
)

__all__ = [
    "Dataset",
    "DatasetAdapter",
    "Records",
    "IncompleteCorpusError",
    "UnresolvedRelevanceError",
    "load_corpus",
    "load_queries",
    "load_qrels",
    "load_metadata",
    "validate_qrels",
]
