"""Usage: uv run python examples/inspect_dataset.py [--local-dir PATH]."""

import argparse

from scientific_tot import Dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--local-dir")
parser.add_argument("--revision", help="Full immutable Hub commit SHA")
args = parser.parse_args()
dataset = Dataset(local_dir=args.local_dir, revision=args.revision)
inventory = dataset.discover()
print("Revision:", inventory.revision or "local/unrecorded")
print("Components:", ", ".join(inventory.components))
for collection in inventory.collections:
    print(collection.key, collection.queries_path)
