"""Frozen, resumable completion suite; preserve every earlier experiment."""
import argparse
import hashlib
import importlib.metadata
from itertools import product
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from mis.anytime import random_fast_anytime
from mis.benchmark import generate_instance, write_json
from mis.completion import prune_weak_constraints, solve_fixed_epochs, solve_timed_control
from mis.controlled import fingerprint
from mis.datasets import load_upstream_dataset
from mis.fast_baselines import fast_degree_greedy, solve_scip_fast
from mis.problem import HypergraphProblem
from mis.verify_report import verify
from check_controlled_protocol import check

SOURCES = ["mis/completion.py", "mis/anytime.py", "mis/controlled.py", "mis/indexed_loss.py",
           "mis/fast_decode.py", "mis/fast_baselines.py", "mis/problem.py", "mis/benchmark.py",
           "mis/datasets.py", "src/models.py", "src/train.py", "scripts/run_completion.py",
           "scripts/check_controlled_protocol.py", "scripts/run_controlled.py"]


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def suite_inputs(plan):
    ablation = []
    for spec in plan["ablation_inputs"]["synthetic"]:
        name = f"n{spec['n']}_k{spec['k']}_d{spec['density']}_s{spec['seed']}"
        instance = {"name": name, **generate_instance(spec["n"], spec["k"], spec["density"], spec["seed"])}
        ablation.extend((name, mode, instance) for mode in spec["definitions"])
    for dataset in plan["ablation_inputs"]["real"]["datasets"]:
        instance = {"name": dataset + "-upstream", **load_upstream_dataset(dataset).to_dict()}
        ablation.extend((dataset, mode, instance) for mode in plan["ablation_inputs"]["real"]["definitions"])
    for group in ("fixed", "timed"):
        for name, mode, instance in ablation:
            yield group, name, mode, instance
    rep = plan["replication"]
    for n, structure, seed in product(rep["sizes"], rep["structures"], rep["instance_seeds"]):
        k, density = structure["k"], structure["density"]
        name = f"n{n}_k{k}_d{density}_s{seed}"
        yield "replication", name, rep["definition"], {"name": name, **generate_instance(n, k, density, seed)}


def variants(plan, group, mode):
    if group == "fixed":
        settings = plan["fixed"]
        result = [{"annealing": a, "penalty": settings["baseline_penalty"]} for a in settings["annealing"]]
        if mode == "weak":
            result.extend({"annealing": "cra", "penalty": p} for p in settings["weak_additional_penalties"])
        return result
    if group == "timed":
        settings = plan["timed"]
        result = [{"annealing": a, "penalty": settings["penalty"], "targets": settings["targets_seconds"]}
                  for a in settings["annealing"]]
        if mode == "weak":
            result.extend({"annealing": "cra", "penalty": settings["penalty"], "targets": [h]}
                          for h in settings["weak_compressed_horizons_seconds"])
        return result
    raise ValueError("Variants apply only to fixed/timed groups")


def variant_key(group, variant, seed):
    penalty = format(float(variant["penalty"]), "g")
    suffix = "" if group == "fixed" else "_h" + format(float(variant["targets"][-1]), "g")
    return f"{variant['annealing']}_p{penalty}{suffix}_seed{seed}"


def structural_report(plan, binding):
    datasets = []
    for dataset in plan["structural"]["datasets"]:
        imported = load_upstream_dataset(dataset)
        reduced, metadata = prune_weak_constraints(imported.problem)
        datasets.append({"dataset": dataset, "original_instance": imported.to_dict(),
                         "reduced_instance": {"n": reduced.n, "edges": reduced.edges}, "certificate": metadata})
    tiny = plan["structural"]["tiny_graph"]
    table = []
    for penalty in plan["structural"]["tiny_penalties"]:
        assignments = []
        for bits in product((0, 1), repeat=tiny["n"]):
            violated = sum(all(bits[i] for i in edge) for edge in tiny["edges"])
            assignments.append({"bits": bits, "size": sum(bits), "feasible": violated == 0,
                                "energy": -sum(bits) + penalty * violated})
        minimum = min(row["energy"] for row in assignments)
        minima = [row for row in assignments if row["energy"] == minimum]
        table.append({"penalty": penalty, "assignments": assignments, "minimum_energy": minimum,
                      "minimizing_bits": [row["bits"] for row in minima],
                      "all_binary_minimizers_feasible": all(row["feasible"] for row in minima)})
    return {"manifest_binding": binding, "weak_redundancy": datasets,
            "tiny_graph": tiny, "penalty_boundary": table,
            "scope": "Boolean weak feasibility only. No reduced graph is used in any training run."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/completion-20261020.json")
    parser.add_argument("--stages", nargs="+", choices=["fixed", "timed", "replication"],
                        default=["fixed", "timed", "replication"])
    parser.add_argument("--max-new-runs", type=int, help="Stop only between complete runs; incomplete output is never a final suite")
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs < 1:
        raise ValueError("max-new-runs must be positive")
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    selection_path = ROOT / plan["selection_source"]
    check(selection_path.parent, selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf8"))
    if selection["chosen"] != plan["locked_configs"]:
        raise ValueError("Original validation selection was not preserved")
    inputs = list(suite_inputs(plan))
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
                   "packages": {p: importlib.metadata.version(p) for p in ["dhg", "numpy", "scipy", "pyscipopt"]},
                   "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()}
    bound = {"plan": plan, "selection": selection, "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
             "source_sha256": source_hashes(), "environment": environment,
             "inputs": {f"{g}/{name}_{mode}": fingerprint(instance) for g, name, mode, instance in inputs}}
    binding = fingerprint(bound)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if manifest["binding"] != binding:
            raise ValueError("Resume refused: source, configuration, selection, input or environment changed")
    else:
        manifest = {**bound, "binding": binding,
                    "git_commit_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        write_json(manifest_path, manifest)
    structural_path = args.output / "structural.json"
    if not structural_path.exists():
        write_json(structural_path, structural_report(plan, binding))
    elif json.loads(structural_path.read_text(encoding="utf8"))["manifest_binding"] != binding:
        raise ValueError("Structural report refers to another frozen suite")

    warm = HypergraphProblem(6, ((0, 1, 2), (2, 3, 4)))
    solve_fixed_epochs(warm, "weak", 0, epochs=2, width=32, checkpoint_every=1)
    solve_timed_control(warm, "weak", 0, annealing="cra", targets=(.02, .04), width=32)
    solve_scip_fast(warm, "weak", .05, cutoff_events=True)
    completed = 0

    def between_runs():
        if source_hashes() != bound["source_sha256"]:
            raise ValueError("Source changed while the frozen experiment was running")
        pause = args.output / ".pause"
        if pause.exists():
            print("PAUSED between complete runs; no measured run is active", flush=True)
            while pause.exists():
                time.sleep(1)
            print("RESUMED", flush=True)
        if args.max_new_runs is not None and completed >= args.max_new_runs:
            print("STOPPED at requested whole-run boundary; suite is not declared complete", flush=True)
            return False
        return True

    previous_group = None
    for group, name, mode, instance in inputs:
        if group not in args.stages:
            continue
        if previous_group is not None and previous_group != group:
            print(f"STAGE_COMPLETE {previous_group}", flush=True)
        previous_group = group
        path = args.output / "cases" / group / f"{name}_{mode}.json"
        write_json(args.output / "instances" / f"{name}.json", instance)
        problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
        if path.exists():
            report = json.loads(path.read_text(encoding="utf8"))
            if report["manifest_binding"] != binding or fingerprint(report["instance"]) != fingerprint(instance):
                raise ValueError("Saved case refers to a different frozen input")
            verify(path)
        else:
            report = {"group": group, "definition": mode, "instance": instance, "manifest_binding": binding,
                      "environment": environment, "runs": [], "records": []}
            if group == "replication":
                start = time.perf_counter()
                selected = fast_degree_greedy(problem, mode)
                report["records"].append({"method": "degree_greedy", "implementation": "fast_equivalent_degree_order",
                                          "selected": selected, "size": len(selected), "feasible": True,
                                          "total_seconds": time.perf_counter() - start})
            write_json(path, report)
        keys = {run["run_key"] for run in report["runs"]}
        if len(keys) != len(report["runs"]):
            raise ValueError("Duplicate whole runs")

        def save_run(key, run, records):
            nonlocal completed
            run["run_key"] = key
            for record in records:
                record["run_key"] = key
            report["runs"].append(run)
            report["records"].extend(records)
            write_json(path, report)
            verify(path)
            keys.add(key)
            completed += 1
            print(f"DONE {group}/{name}_{mode} {key}: " + ", ".join(
                f"{r.get('budget_seconds', 'fixed')}={r['size']}" for r in records), flush=True)

        if group in ("fixed", "timed"):
            for seed in plan["model_seeds"]:
                for variant in variants(plan, group, mode):
                    key = variant_key(group, variant, seed)
                    if key in keys:
                        continue
                    if not between_runs():
                        return
                    if group == "fixed":
                        run = solve_fixed_epochs(problem, mode, seed, epochs=plan["fixed"]["epochs"], width=plan["width"],
                                                 penalty=variant["penalty"], annealing=variant["annealing"],
                                                 checkpoint_every=plan["fixed"]["checkpoint_every"])
                        record = {**run["best"], "method": "completion_fixed_epochs", "seed": seed,
                                  "total_seconds": run["run_total_seconds"], "epochs_completed": run["epochs_completed"],
                                  "selection_policy": "first_maximum_epoch_candidate", "variant": variant}
                        records = [record]
                    else:
                        run = solve_timed_control(problem, mode, seed, annealing=variant["annealing"], penalty=variant["penalty"],
                                                  targets=variant["targets"], interval=plan["candidate_interval_seconds"], width=plan["width"])
                        records = run["records"]
                    run["variant"] = variant
                    save_run(key, run, records)
            expected = len(plan["model_seeds"]) * len(variants(plan, group, mode))
        else:
            seconds = plan["replication"]["seconds"]
            if "scip_30" not in keys:
                if not between_runs():
                    return
                run = solve_scip_fast(problem, mode, seconds, cutoff_events=True)
                cutoff = run["cutoff"]
                record = {**cutoff, "method": run["method"], "budget_seconds": seconds, "cutoff_seconds": seconds,
                          "total_seconds": run["total_seconds"], "snapshot_emitted_seconds": run["total_seconds"],
                          "best_available_seconds": cutoff["available_seconds"], "completion_status": run["status"],
                          "completion_size": run["size"], "implementation": run["implementation"]}
                save_run("scip_30", run, [record])
            for seed in plan["model_seeds"]:
                key = f"neural_{seed}"
                if key not in keys:
                    if not between_runs():
                        return
                    run = solve_timed_control(problem, mode, seed, annealing=plan["replication"]["neural_annealing"],
                                              penalty=plan["replication"]["penalty"], targets=[seconds],
                                              interval=plan["candidate_interval_seconds"], width=plan["width"])
                    save_run(key, run, run["records"])
                key = f"random_{seed}"
                if key not in keys:
                    if not between_runs():
                        return
                    run = random_fast_anytime(problem, mode, seed, targets=[seconds])
                    save_run(key, run, [run["single"], *run["records"]])
            expected = 1 + 2 * len(plan["model_seeds"])
        if len(report["runs"]) != expected:
            raise ValueError("Incomplete or unexpected case runs")
    if previous_group is not None:
        print(f"STAGE_COMPLETE {previous_group}", flush=True)
    if source_hashes() != bound["source_sha256"]:
        raise ValueError("Source changed during the suite")
    print("RUNNER_COMPLETE requested stages saved; independent full-suite audit is required", flush=True)


if __name__ == "__main__":
    main()
