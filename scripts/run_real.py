"""Evaluate all three audited upstream data sets with the locked configuration."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mis.benchmark import random_decoder, write_json
from mis.controlled import bind_manifest, fingerprint, solve_controlled
from mis.datasets import load_upstream_dataset
from mis.problem import HypergraphProblem, degree_greedy
from mis.reference import solve_scip
from mis.verify_report import verify
from run_controlled import expected, identity
from check_controlled_protocol import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=ROOT / "results/controlled-validation-20261018")
    parser.add_argument("--selection", type=Path, default=ROOT / "results/controlled-validation-20261018/selection.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    validation = check(args.validation, args.selection)
    selection = json.loads(args.selection.read_text(encoding="utf8"))
    plan = validation["protocol"]["plan"]
    settings = {"model_seeds": [0, 1, 2]}
    datasets = {name: load_upstream_dataset(name).to_dict() for name in ["cora", "high", "primary"]}
    protocol = {"phase": "real", "plan": plan, "selection": selection,
                "datasets": {name: fingerprint(instance) for name, instance in datasets.items()},
                "model_seeds": settings["model_seeds"], "definitions": plan["definitions"],
                "additional_source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                             for path in ["mis/datasets.py", "scripts/run_real.py", "scripts/check_controlled_protocol.py"]},
                "limits": "Three public upstream versions, no tuning on their outcomes; data identities and redundancy caveats in docs/data-provenance.md."}
    warm = HypergraphProblem(6, ((0, 1, 2), (3, 4)))
    solve_controlled(warm, "weak", "w256_cra", 0, epochs=2)
    solve_scip(warm, "weak", 0.1, total_budget=True)
    manifest = bind_manifest(args.output, protocol)
    for name, instance in datasets.items():
        instance = {"name": name + "-upstream", **instance}
        write_json(args.output / "instances" / f"{name}-upstream.json", instance)
        problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
        for mode in plan["definitions"]:
            path = args.output / "cases" / f"{name}_{mode}.json"
            if path.exists():
                report = json.loads(path.read_text(encoding="utf8"))
                if report["manifest_binding"] != manifest["binding"] or fingerprint(report["instance"]) != fingerprint(instance):
                    raise ValueError("Resume refused: real instance or source binding differs")
                verify(path)
            else:
                report = {"definition": mode, "instance": instance, "records": [], "purpose": "real",
                          "manifest_binding": manifest["binding"], "environment": manifest["environment"]}
            required = expected("test", settings, mode, selection)
            existing = [identity(r) for r in report["records"]]
            if len(set(existing)) != len(existing) or any(item not in required for item in existing):
                raise ValueError("Unexpected or duplicate real records")
            for method, config, seed in required:
                if (method, config, seed) in existing:
                    continue
                if method == "degree_greedy":
                    start = time.perf_counter()
                    chosen = degree_greedy(problem, mode)
                    record = {"method": method, "selected": chosen, "size": len(chosen),
                              "feasible": True, "total_seconds": time.perf_counter() - start}
                elif method == "scip_milp_greedy_start":
                    record = solve_scip(problem, mode, plan["budget_seconds"], total_budget=True)
                elif method.startswith("random_"):
                    budget = plan["budget_seconds"] if "restart" in method else None
                    record = random_decoder(problem, mode, seed, budget)
                else:
                    record = solve_controlled(problem, mode, config, seed, budget=plan["budget_seconds"])
                report["records"].append(record)
                write_json(path, report)
                verify(path)
                print(f"REAL {name} {mode} {method} {config} seed={seed} size={record['size']} "
                      f"seconds={record['total_seconds']:.3f} epochs={record.get('epochs_completed', '-')}", flush=True)


if __name__ == "__main__":
    main()
