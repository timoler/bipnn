"""Compare unchanged deterministic decoder strategies on pinned real data.

Run separately after neural training has stopped. Both methods include complete
decoding and, for the fast version, per-call incidence-index construction. This
is a CPU implementation benchmark, not a new optimization heuristic or evidence
that any previously frozen neural experiment used the faster implementation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source_hashes():
    return {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
        "mis/fast_decode.py", "mis/problem.py", "mis/controlled.py", "mis/datasets.py",
        "scripts/benchmark_decode.py",
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["cora", "high", "primary"],
                        default=["cora", "high", "primary"])
    parser.add_argument("--definitions", nargs="+", choices=["weak", "strong"], default=["weak", "strong"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups must be nonnegative")
    for key in ["datasets", "definitions", "seeds"]:
        values = getattr(args, key)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate {key} are not supported")
    if args.output.exists():
        parser.error("Output directory already exists; use a fresh directory")
    # These settings affect this benchmark process only. Both algorithms use
    # serial Python code; no GPU device or GPU tensor is created here.
    thread_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
    for name in thread_vars:
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from mis.controlled import decode_stages
    from mis.datasets import load_upstream_dataset
    from mis.fast_decode import fast_decode_stages

    methods = [("reference_decode_stages", decode_stages),
               ("incidence_heap_decode_stages", fast_decode_stages)]
    imported = {name: load_upstream_dataset(name) for name in args.datasets}
    args.output.mkdir(parents=True, exist_ok=False)
    instance_files = {}
    for name, data in imported.items():
        relative = f"instances/{name}.json"
        path = args.output / relative
        write_new_json(path, data.to_dict())
        instance_files[name] = {"path": relative,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "raw_sha256": data.metadata["raw_sha256"],
                                "lf_normalized_sha256": data.metadata["lf_normalized_sha256"]}
    manifest = {
        "schema_version": 1,
        "protocol": {
            "datasets": args.datasets, "definitions": args.definitions, "seeds": args.seeds,
            "repeats_per_method": args.repeats, "warmups_per_method": args.warmups,
            "scores": "random.Random(seed).random(), identical list supplied to both methods",
            "score_reuse": "same dataset and seed reuse the same scores across weak/strong",
            "method_order": "alternate reference-first and fast-first on each repeat; initial order alternates by case",
            "timing": "perf_counter wall seconds and process_time CPU seconds; includes index construction on every fast call",
            "warmup_timing": "warmups excluded from trial timing; no index or solution is reused",
            "verification": "entire per-stage output dictionaries must match exactly on every repetition",
            "cpu_threads": 1, "gpu_used": False,
            "scope": "implementation-only benchmark; no change to frozen neural experiment decoder",
        },
        "source_sha256": source_hashes(),
        "git_commit_at_start": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {name: os.environ[name] for name in thread_vars},
            "timing_conditions": "ordinary desktop process; CPU affinity and background workloads are not controlled",
        },
        "instances": instance_files,
    }
    write_new_json(args.output / "manifest.json", manifest)
    summaries = []
    case_index = 0
    for name in args.datasets:
        problem = imported[name].problem
        for mode in args.definitions:
            for seed in args.seeds:
                case_started = time.perf_counter()
                generator = random.Random(seed)
                scores = [generator.random() for _ in range(problem.n)]
                expected = None
                for _ in range(args.warmups):
                    for _, decoder in methods:
                        result = decoder(problem, scores, mode)
                        if expected is None:
                            expected = result
                        elif result != expected:
                            raise AssertionError("Decoders differ during warmup")
                trials = []
                for repeat in range(args.repeats):
                    order = methods if (case_index + repeat) % 2 == 0 else list(reversed(methods))
                    for order_index, (method_name, decoder) in enumerate(order):
                        cpu_started = time.process_time()
                        wall_started = time.perf_counter()
                        result = decoder(problem, scores, mode)
                        wall_seconds = time.perf_counter() - wall_started
                        cpu_seconds = time.process_time() - cpu_started
                        if expected is None:
                            expected = result
                        if result != expected:
                            raise AssertionError("Per-stage decoder outputs are not identical")
                        if not problem.feasible(result["selected"], mode):
                            raise AssertionError("A decoder returned an infeasible solution")
                        trials.append({"method": method_name, "repeat": repeat,
                                       "order_in_repeat": order_index, "wall_seconds": wall_seconds,
                                       "cpu_seconds": cpu_seconds, "matches_reference": True,
                                       **result})
                timing = {}
                for method_name, _ in methods:
                    values = [r["wall_seconds"] for r in trials if r["method"] == method_name]
                    timing[method_name] = {"trials": len(values), "wall_seconds": values,
                                           "mean_seconds": statistics.mean(values),
                                           "median_seconds": statistics.median(values),
                                           "min_seconds": min(values), "max_seconds": max(values)}
                fast_seconds = timing[methods[1][0]]["median_seconds"]
                speedup = timing[methods[0][0]]["median_seconds"] / fast_seconds
                case_id = f"{name}-{mode}-seed{seed}"
                case = {"case_id": case_id, "dataset": name, "definition": mode, "seed": seed,
                        "instance": instance_files[name], "scores": scores, "trials": trials,
                        "timing": timing, "median_speedup_reference_over_fast": speedup,
                        "all_stages_identical": True}
                relative = f"cases/{case_id}.json"
                write_new_json(args.output / relative, case)
                summaries.append({"case_id": case_id, "dataset": name, "definition": mode, "seed": seed,
                                  "case_file": relative, "n": problem.n, "edges": len(problem.edges),
                                  "raw_size": expected["raw_size"], "repaired_size": expected["repaired_size"],
                                  "size": expected["size"], "timing": timing,
                                  "median_speedup_reference_over_fast": speedup,
                                  "all_stages_identical": True})
                print(json.dumps({"case": case_id, "size": expected["size"], "identical": True,
                                  "median_speedup": speedup,
                                  "case_seconds": time.perf_counter() - case_started}), flush=True)
                case_index += 1
    if source_hashes() != manifest["source_sha256"]:
        raise RuntimeError("Bound source files changed during the benchmark")
    speedups = [r["median_speedup_reference_over_fast"] for r in summaries]
    summary = {"schema_version": 1, "complete": True, "case_count": len(summaries),
               "timed_decode_calls": len(summaries) * args.repeats * len(methods),
               "all_stages_identical": True, "cases": summaries,
               "speedup_scope": "descriptive CPU decoding times on this desktop; not neural solve speedup",
               "speedup_median_across_cases": statistics.median(speedups),
               "speedup_geometric_mean_across_cases": math.exp(statistics.mean(map(math.log, speedups))),
               "speedup_min": min(speedups), "speedup_max": max(speedups)}
    write_new_json(args.output / "summary.json", summary)
    print(json.dumps({"complete": True, "cases": len(summaries),
                      "timed_decode_calls": summary["timed_decode_calls"],
                      "all_stages_identical": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
