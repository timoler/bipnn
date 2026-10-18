"""Fixed fast-decoder protocol: atomic whole trajectories and strict past prefixes."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mis.anytime import random_fast_anytime, solve_fast_anytime
from mis.benchmark import generate_instance, write_json
from mis.controlled import fingerprint
from mis.datasets import load_upstream_dataset
from mis.fast_baselines import fast_degree_greedy, solve_scip_fast
from mis.problem import HypergraphProblem
from mis.verify_report import verify
from check_controlled_protocol import check

SOURCES = ["mis/anytime.py", "mis/controlled.py", "mis/indexed_loss.py", "mis/fast_decode.py",
           "mis/fast_baselines.py", "mis/problem.py", "mis/benchmark.py", "mis/datasets.py",
           "src/models.py", "src/train.py", "scripts/run_anytime.py", "scripts/check_controlled_protocol.py",
           "scripts/run_controlled.py"]


def sources():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def inputs(plan):
    for spec in plan["synthetic"]:
        name = f"n{spec['n']}_k{spec['k']}_d{spec['density']}_s{spec['seed']}"
        instance = {"name": name, **generate_instance(spec["n"], spec["k"], spec["density"], spec["seed"])}
        for mode in spec["definitions"]:
            yield name, mode, instance
    for dataset in plan["real"]["datasets"]:
        instance = {"name": dataset + "-upstream", **load_upstream_dataset(dataset).to_dict()}
        for mode in plan["real"]["definitions"]:
            yield dataset, mode, instance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/fast-anytime-20261018.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    selection_path = ROOT / plan["selection_source"]
    check(selection_path.parent, selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf8"))
    if selection["chosen"] != plan["locked_configs"]:
        raise ValueError("Protocol configuration differs from locked validation choice")
    all_inputs = list(inputs(plan))
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
                   "packages": {p: importlib.metadata.version(p) for p in ["dhg", "numpy", "scipy", "pyscipopt"]},
                   "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()}
    bound = {"plan": plan, "selection": selection,
             "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
             "source_sha256": sources(), "environment": environment,
             "inputs": {f"{name}_{mode}": fingerprint(instance) for name, mode, instance in all_inputs}}
    manifest_path = args.output / "manifest.json"
    binding = fingerprint(bound)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if manifest["binding"] != binding:
            raise ValueError("Resume refused: sources, input, environment or protocol changed")
    else:
        manifest = {**bound, "binding": binding,
                    "git_commit_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        write_json(manifest_path, manifest)
    warm = HypergraphProblem(6, ((0, 1, 2), (2, 3, 4)))
    solve_fast_anytime(warm, "weak", "w1024_cra", 0, targets=[.02, .04])
    solve_scip_fast(warm, "weak", .05, cutoff_events=True)
    targets = plan["checkpoints_seconds"]
    for name, mode, instance in all_inputs:
        if sources() != bound["source_sha256"]:
            raise ValueError("Source changed during experiment")
        path = args.output / "cases" / f"{name}_{mode}.json"
        write_json(args.output / "instances" / f"{name}.json", instance)
        problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
        if path.exists():
            report = json.loads(path.read_text(encoding="utf8"))
            if report["manifest_binding"] != binding or fingerprint(report["instance"]) != fingerprint(instance):
                raise ValueError("Saved case binding or instance changed")
            verify(path)
        else:
            begin = time.perf_counter()
            selected = fast_degree_greedy(problem, mode)
            report = {"definition": mode, "instance": instance, "manifest_binding": binding,
                      "environment": environment, "runs": [], "records": [
                          {"method": "degree_greedy", "implementation": "fast_equivalent_degree_order",
                           "selected": selected, "size": len(selected), "feasible": True,
                           "total_seconds": time.perf_counter() - begin}]}
            write_json(path, report)
        keys = [r["run_key"] for r in report["runs"]]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate seed trajectories")

        def save_run(key, run, records):
            run["run_key"] = key
            for record in records:
                record["run_key"] = key
            report["runs"].append(run)
            report["records"].extend(records)
            # Whole trajectory and all snapshots enter one atomic replacement.
            write_json(path, report)
            verify(path)
            print(f"DONE {name} {mode} {key}: " + ", ".join(
                f"{r.get('budget_seconds', 'single')}s={r['size']}" for r in records), flush=True)

        for budget in targets:
            key = f"scip_{budget}"
            if key in keys:
                continue
            run = solve_scip_fast(problem, mode, budget, cutoff_events=True)
            cutoff = run["cutoff"]
            record = {**cutoff, "method": run["method"], "budget_seconds": budget,
                      "cutoff_seconds": budget, "total_seconds": run["total_seconds"],
                      "snapshot_emitted_seconds": run["total_seconds"],
                      "best_available_seconds": cutoff["available_seconds"],
                      "completion_status": run["status"], "completion_size": run["size"],
                      "implementation": run["implementation"]}
            save_run(key, run, [record])
        for seed in plan["model_seeds"]:
            key = f"neural_{seed}"
            if key not in keys:
                run = solve_fast_anytime(problem, mode, plan["locked_configs"][mode], seed,
                                         targets=targets, interval=plan["candidate_interval_seconds"])
                save_run(key, run, run["records"])
            key = f"random_{seed}"
            if key not in keys:
                run = random_fast_anytime(problem, mode, seed, targets=targets)
                save_run(key, run, [run["single"], *run["records"]])
        if len(report["records"]) != 1 + len(targets) + len(plan["model_seeds"]) * (1 + 2 * len(targets)):
            raise ValueError("Case is incomplete or contains extra records")
    if sources() != bound["source_sha256"]:
        raise ValueError("Source changed during experiment")
    print("COMPLETE: all trajectories and strict-prefix snapshots saved", flush=True)


if __name__ == "__main__":
    main()
