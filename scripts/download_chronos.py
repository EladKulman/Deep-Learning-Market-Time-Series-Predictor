#!/usr/bin/env python3
"""Pre-stage the tested base model before running on an offline Slurm compute node."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

TESTED_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--revision", default=TESTED_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    args = parser.parse_args()
    location = snapshot_download(repo_id=args.model_id, revision=args.revision, cache_dir=args.cache_dir,
                                 allow_patterns=["*.json", "*.safetensors"])
    manifest = {"model_id": args.model_id, "revision": Path(location).name, "path": location}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "chronos_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
