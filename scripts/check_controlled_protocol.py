"""Cross-stage guard, including current sources/environment and locked selection."""
import argparse
import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mis.controlled import fingerprint
from run_controlled import select


def check(validation, selection_path=None):
    manifest = json.loads((Path(validation) / "manifest.json").read_text(encoding="utf8"))
    bound = {"protocol": manifest["protocol"], "source_sha256": manifest["source_sha256"]}
    if fingerprint(bound) != manifest["binding"]:
        raise ValueError("Manifest binding does not match contents")
    for path, expected_hash in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Current source differs from validation: {path}")
    current = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               "cpu_threads": torch.get_num_threads()}
    if current != manifest["environment"]:
        raise ValueError("Current environment differs from validation")
    if selection_path is not None:
        selection = json.loads(Path(selection_path).read_text(encoding="utf8"))
        if selection["validation_binding"] != manifest["binding"]:
            raise ValueError("Selection refers to a different validation manifest")
        # Independently re-check complete validation and the deterministic rule;
        # select() refuses to replace a different already locked selection.
        select(SimpleNamespace(output=Path(validation), selection=Path(selection_path)), manifest["protocol"]["plan"])
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("validation", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    check(args.validation, args.selection)
    print("PASS: source, environment, manifest and selection checks")
