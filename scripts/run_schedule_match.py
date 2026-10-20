"""Independent matched-pre15 sampling supplement; parent results stay frozen.

Use --validate-only for CPU metadata/API checks without a CUDA context or output
writes. Formal execution must be scheduled after the parent runner exits; the
parent summary is deliberately not a dependency of this child batch.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from mis.benchmark import write_json
from mis.problem import HypergraphProblem
from mis.verify_report import verify

RUNNER_PATH = "scripts/run_schedule_match.py"


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def raw_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def actual_environment(parent_environment):
    """Read live metadata without initializing a Torch CUDA context."""
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True).strip().splitlines()
    if len(gpu_rows) != 1:
        raise ValueError("This registered single-GPU protocol requires exactly one visible physical GPU")
    gpu, driver = [part.strip() for part in gpu_rows[0].split(",", 1)]
    environment = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": gpu, "cpu_threads": torch.get_num_threads(),
        "packages": {name: importlib.metadata.version(name)
                     for name in parent_environment["packages"]},
        "driver": driver,
    }
    if environment != parent_environment:
        raise ValueError("Current Python/packages/CUDA build/GPU/driver/threads differ from the parent environment")
    return environment


def nominal_calendar(targets, interval):
    horizon = targets[-1]
    due = sorted({i * interval for i in range(1, math.ceil(horizon / interval))} |
                 {target - min(.1, target / 10) for target in targets})
    return due


def registered_input_names(parent_plan):
    names = []
    for spec in parent_plan["ablation_inputs"]["synthetic"]:
        if "weak" in spec["definitions"]:
            names.append(f"n{spec['n']}_k{spec['k']}_d{spec['density']}_s{spec['seed']}")
    real = parent_plan["ablation_inputs"]["real"]
    if "weak" in real["definitions"]:
        names.extend(real["datasets"])
    return names


def validate_plan(plan, parent):
    if plan["parent_manifest_binding"] != parent["binding"]:
        raise ValueError("The child plan is registered against another parent manifest")
    if plan["definition"] != "weak" or plan["model_seeds"] != parent["plan"]["model_seeds"]:
        raise ValueError("Definitions/seeds differ from the registered weak ablation subset")
    if plan["variant"] != {"annealing": "cra", "penalty": 2, "targets": [5, 15],
                           "sampling_control": "matched_pre15_calendar"}:
        raise ValueError("Unexpected schedule-match variant")
    if (plan["width"] != 1024 or plan["width"] != parent["plan"]["width"] or
            plan["candidate_interval_seconds"] != parent["plan"]["candidate_interval_seconds"] or
            plan["comparison_cutoff_seconds"] != 15):
        raise ValueError("Width, sampling interval or comparison cutoff changed")
    names = [row["name"] for row in plan["inputs"]]
    if names != registered_input_names(parent["plan"]) or len(names) != 5:
        raise ValueError("Inputs must be all five prespecified weak ablation instances in parent order")
    if plan["expected_counts"] != {"cases": 5, "neural_runs": 15, "primary_records": 30}:
        raise ValueError("Unexpected registered case/run/record counts")
    initial = plan["paired_initial_outputs"]
    if (initial["state_and_feature_hashes"] != "exact equality" or
            initial["score_relative_tolerance"] != 1e-5 or initial["score_absolute_tolerance"] != 1e-5):
        raise ValueError("Initial state/feature hash policy or first-forward numerical tolerance changed")
    if parent["plan"]["timed"]["targets_seconds"] != [5, 15, 30]:
        raise ValueError("The parent reference horizon no longer matches fixed30")
    cutoff, interval = plan["comparison_cutoff_seconds"], plan["candidate_interval_seconds"]
    child_due = [value for value in nominal_calendar(plan["variant"]["targets"], interval) if value < cutoff]
    parent_due = [value for value in nominal_calendar(parent["plan"]["timed"]["targets_seconds"], interval)
                  if value < cutoff]
    child_snapshots = [value for value in plan["variant"]["targets"] if value < cutoff]
    parent_snapshots = [value for value in parent["plan"]["timed"]["targets_seconds"] if value < cutoff]
    if child_due != parent_due or child_snapshots != parent_snapshots:
        raise ValueError("Nominal sampling/snapshot calendars strictly before the cutoff are not matched")


def prepare_binding(parent_dir, plan_path):
    """Validate current inputs/source/environment; no case outcomes are read."""
    parent_path = parent_dir / "manifest.json"
    parent = read_json(parent_path)
    parent_bound = {key: value for key, value in parent.items()
                    if key not in ("binding", "git_commit_at_start")}
    if fingerprint(parent_bound) != parent["binding"]:
        raise ValueError("Parent manifest binding cannot be recomputed")
    plan = read_json(plan_path)
    validate_plan(plan, parent)
    current_parent_plan = ROOT / plan["parent_plan"]
    if read_json(current_parent_plan) != parent["plan"]:
        raise ValueError("Current parent plan differs from its frozen manifest")
    source = {}
    for name, expected in parent["source_sha256"].items():
        actual = raw_sha(ROOT / name)
        if actual != expected:
            raise ValueError(f"Frozen parent source hash differs: {name}")
        source[name] = actual
    source[RUNNER_PATH] = raw_sha(ROOT / RUNNER_PATH)
    selection_path = ROOT / parent["plan"]["selection_source"]
    if (raw_sha(selection_path) != parent["selection_sha256"] or
            read_json(selection_path) != parent["selection"]):
        raise ValueError("Frozen parent validation selection changed")
    inputs, input_hashes, file_hashes = [], {}, {}
    for spec in plan["inputs"]:
        name = spec["name"]
        relative_path = f"instances/{name}.json"
        if spec["parent_instance"] != relative_path:
            raise ValueError("Input filename differs from its registered parent instance")
        path = parent_dir / relative_path
        instance = read_json(path)
        signature = fingerprint(instance)
        if signature != spec["fingerprint"]:
            raise ValueError(f"Registered full input fingerprint differs: {name}")
        for group in ("fixed", "timed"):
            if parent["inputs"].get(f"{group}/{name}_weak") != signature:
                raise ValueError(f"Parent input binding differs: {group}/{name}_weak")
        HypergraphProblem(instance["n"], tuple(tuple(edge) for edge in instance["edges"]))
        inputs.append((name, instance))
        input_hashes[f"schedule_match/{name}_weak"] = signature
        file_hashes[relative_path] = raw_sha(path)
    environment = actual_environment(parent["environment"])
    bound = {
        "parent_manifest_binding": parent["binding"],
        "parent_manifest_sha256": raw_sha(parent_path),
        "parent_plan_sha256": raw_sha(current_parent_plan),
        "plan": plan, "plan_sha256": raw_sha(plan_path), "source_sha256": source,
        "selection_sha256": raw_sha(selection_path), "environment": environment,
        "inputs": input_hashes, "input_file_sha256": file_hashes,
    }
    return bound, inputs


def run_key(seed):
    return f"cra_p2_h15_matched_seed{seed}"


def validate_saved_case(report, name, instance, manifest):
    if (report.get("group") != "schedule_match" or report.get("definition") != "weak" or
            report.get("manifest_binding") != manifest["binding"] or
            report.get("parent_manifest_binding") != manifest["parent_manifest_binding"] or
            report.get("environment") != manifest["environment"] or
            fingerprint(report["instance"]) != fingerprint(instance)):
        raise ValueError(f"Saved child case differs from its bound protocol/input: {name}")
    expected = {run_key(seed): seed for seed in manifest["plan"]["model_seeds"]}
    keys = [run["run_key"] for run in report["runs"]]
    if len(set(keys)) != len(keys) or any(key not in expected for key in keys):
        raise ValueError("Duplicate or unregistered child whole run")
    flattened = []
    for run in report["runs"]:
        key = run["run_key"]
        if run["seed"] != expected[key] or run["variant"] != manifest["plan"]["variant"]:
            raise ValueError("Saved child run seed/variant changed")
        settings = run["settings"]
        if (settings["targets_seconds"] != [5., 15.] or settings["horizon_seconds"] != 15. or
                settings["width"] != 1024 or settings["annealing"] != "cra" or settings["penalty"] != 2 or
                settings["candidate_interval_seconds"] != manifest["plan"]["candidate_interval_seconds"]):
            raise ValueError("Saved child whole-run settings changed")
        if ([record["budget_seconds"] for record in run["records"]] != [5., 15.] or
                any(record.get("run_key") != key for record in run["records"])):
            raise ValueError("A saved whole run must contain both original snapshots")
        flattened.extend(run["records"])
    if flattened != report["records"]:
        raise ValueError("Child case records are not exactly its saved whole-run snapshots")
    return set(keys)


def parent_reference_runs(parent_dir, inputs, seeds, parent_binding, environment):
    """Require completed parent fixed30 references; no summary dependency."""
    references = {}
    for name, instance in inputs:
        path = parent_dir / "cases" / "timed" / f"{name}_weak.json"
        report = read_json(path)
        if (fingerprint(report["instance"]) != fingerprint(instance) or report["group"] != "timed" or
                report["definition"] != "weak" or report["manifest_binding"] != parent_binding or
                report["environment"] != environment):
            raise ValueError(f"Parent timed input/group mismatch: {name}")
        by_key = {run["run_key"]: run for run in report["runs"]}
        if len(by_key) != len(report["runs"]):
            raise ValueError("Duplicate parent timed whole runs")
        required = {f"{variant}_seed{seed}" for seed in seeds
                    for variant in ("cra_p2_h30", "none_p2_h30", "cra_p2_h5", "cra_p2_h15")}
        if set(by_key) != required or len(report["records"]) != 8 * len(seeds):
            raise ValueError(f"Parent weak timed case is not yet complete: {name}")
        verify(path)
        for seed in seeds:
            key = f"cra_p2_h30_seed{seed}"
            if key not in by_key:
                raise ValueError(f"Parent fixed30 reference is not yet saved: {name}/{key}")
            run = by_key[key]
            if (run["settings"]["targets_seconds"] != [5., 15., 30.] or
                    run["settings"]["horizon_seconds"] != 30. or
                    run["variant"] != {"annealing": "cra", "penalty": 2, "targets": [5, 15, 30]} or
                    [record["budget_seconds"] for record in run["records"]] != [5., 15., 30.]):
                raise ValueError("Parent fixed30 reference settings/snapshots changed")
            references[(name, seed)] = run
    return references


def compare_initial_state(child, parent):
    for field in ("initial_state_sha256", "feature_sha256"):
        if child[field] != parent[field]:
            raise ValueError(f"Paired child/parent initial state differs: {field}")
    child_first = child["candidates"][child["initial_index"]]
    parent_first = parent["candidates"][parent["initial_index"]]
    if (len(child_first["scores"]) != len(parent_first["scores"]) or
            any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                for a, b in zip(child_first["scores"], parent_first["scores"]))):
        raise ValueError("Paired child/parent first pre-update scores differ beyond numerical tolerance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=ROOT / "results/completion-20261020")
    parser.add_argument("--output", type=Path, default=ROOT / "results/schedule-match-20261020")
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/schedule-match-20261020.json")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only inspect CPU metadata/API/bindings; no CUDA context, warmup or output writes")
    parser.add_argument("--max-new-runs", type=int, help="Stop only between complete runs")
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs < 1:
        raise ValueError("max-new-runs must be positive")
    if args.parent.resolve() == args.output.resolve():
        raise ValueError("The child output must not be the frozen parent result directory")
    torch.set_num_threads(1)
    bound, inputs = prepare_binding(args.parent, args.plan)
    binding = fingerprint(bound)
    if args.validate_only:
        # Read the callable signature without invoking a solver or initializing CUDA.
        from inspect import signature
        from mis.completion import solve_fixed_epochs, solve_timed_control
        signature(solve_fixed_epochs).bind(HypergraphProblem(6, ()), "weak", 0,
                                           epochs=3, width=32, checkpoint_every=1)
        signature(solve_timed_control).bind(HypergraphProblem(6, ()), "weak", 0,
                                           annealing="cra", penalty=2., targets=[5, 15], interval=1., width=1024)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only validation unexpectedly initialized a Torch CUDA context")
        print(json.dumps({"validation": "CPU metadata/API/bindings only; no output writes or training",
                          "binding_if_started_now": binding, "parent_manifest_binding": bound["parent_manifest_binding"],
                          "source_files_matched_to_parent": len(bound["source_sha256"]) - 1,
                          "input_names": [name for name, _ in inputs], "environment": bound["environment"],
                          "expected_counts": bound["plan"]["expected_counts"],
                          "torch_cuda_context_initialized": torch.cuda.is_initialized()}, indent=2), flush=True)
        return

    references = parent_reference_runs(args.parent, inputs, bound["plan"]["model_seeds"],
                                       bound["parent_manifest_binding"], bound["environment"])
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["binding"] != binding or any(manifest.get(key) != value for key, value in bound.items()):
            raise ValueError("Resume refused: source, plan, parent, environment or input changed")
    else:
        manifest = {**bound, "binding": binding,
                    "git_commit_at_start": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(manifest_path, manifest)

    reports, all_complete = {}, True
    for name, instance in inputs:
        path = args.output / "cases" / f"{name}_weak.json"
        if path.exists():
            report = read_json(path)
            keys = validate_saved_case(report, name, instance, manifest)
            verify(path)
        else:
            report = {"group": "schedule_match", "definition": "weak", "instance": instance,
                      "parent_manifest_binding": bound["parent_manifest_binding"],
                      "manifest_binding": binding, "environment": bound["environment"],
                      "runs": [], "records": []}
            keys = set()
        reports[name] = (path, report, keys)
        all_complete = all_complete and len(keys) == len(bound["plan"]["model_seeds"])
    if all_complete:
        print("SCHEDULE_MATCH_ALREADY_SAVED; independent joint audit still required", flush=True)
        return

    if torch.cuda.get_device_name() != bound["environment"]["gpu"]:
        raise ValueError("Torch selected a different GPU than the registered parent")
    from mis.completion import solve_fixed_epochs, solve_timed_control
    warm = HypergraphProblem(6, ((0, 1, 2), (2, 3, 4)))
    solve_fixed_epochs(warm, "weak", 0, epochs=3, width=32, checkpoint_every=1)
    solve_timed_control(warm, "weak", 0, annealing="cra", targets=(.02, .04), width=32)
    completed = 0
    for name, instance in inputs:
        path, report, keys = reports[name]
        problem = HypergraphProblem(instance["n"], tuple(tuple(edge) for edge in instance["edges"]))
        for seed in bound["plan"]["model_seeds"]:
            key = run_key(seed)
            if key in keys:
                continue
            if args.max_new_runs is not None and completed >= args.max_new_runs:
                print("STOPPED at whole-run boundary; supplement not declared complete", flush=True)
                return
            current, _ = prepare_binding(args.parent, args.plan)
            if current != bound:
                raise ValueError("Bound source/plan/environment/input changed between complete child runs")
            variant = bound["plan"]["variant"]
            run = solve_timed_control(problem, "weak", seed, annealing=variant["annealing"],
                                      penalty=variant["penalty"], targets=variant["targets"],
                                      interval=bound["plan"]["candidate_interval_seconds"], width=bound["plan"]["width"])
            compare_initial_state(run, references[(name, seed)])
            run["run_key"], run["variant"] = key, dict(variant)
            for record in run["records"]:
                record["run_key"] = key
            report["runs"].append(run)
            report["records"].extend(run["records"])
            validate_saved_case(report, name, instance, manifest)
            write_json(path, report)
            verify(path)
            keys.add(key)
            completed += 1
            print(f"DONE schedule_match/{name}_weak {key}: " + ", ".join(
                f"{record['budget_seconds']}={record['size']}" for record in run["records"]), flush=True)
    current, _ = prepare_binding(args.parent, args.plan)
    if current != bound:
        raise ValueError("Bound source/plan/environment/input changed during child execution")
    print("SCHEDULE_MATCH_RUNNER_COMPLETE: 15 whole runs and 30 snapshots saved; independent joint audit required", flush=True)


if __name__ == "__main__":
    main()
