"""Fixed synthetic suite with resumable, independently auditable JSON reports."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .problem import HypergraphProblem, degree_greedy, repair_and_extend


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def generate_instance(n, k, density, seed):
    import math
    m = n * density
    if k > n or m > math.comb(n, k):
        raise ValueError("Requested more distinct edges than possible")
    rng = random.Random(seed)
    edges = set()
    while len(edges) < m:
        edges.add(tuple(sorted(rng.sample(range(n), k))))
    return {"n": n, "edges": sorted(edges),
            "generator": {"family": "uniform_fixed_edge_count", "k": k,
                          "density_m_over_n": density, "seed": seed}}


def random_decoder(problem, definition, seed, budget=None):
    rng = random.Random(seed)
    start = time.perf_counter()
    best, best_raw, best_scores = None, None, None
    trials = 0
    while True:
        scores = [rng.random() for _ in range(problem.n)]
        raw = [i for i, p in enumerate(scores) if p >= 0.5]
        selected = repair_and_extend(problem, scores, definition)
        trials += 1
        if best is None or len(selected) > len(best):
            best, best_raw, best_scores = selected, raw, scores
        if budget is None or time.perf_counter() - start >= budget:
            break
    return {"method": "random_score_same_decoder" if budget is None else "random_restart_same_decoder",
            "seed": seed, "selected": best, "size": len(best), "feasible": True,
            "raw_selected": best_raw, "raw_size": len(best_raw),
            "raw_feasible": problem.feasible(best_raw, definition), "scores": best_scores,
            "total_seconds": time.perf_counter() - start,
            "trials": trials, "time_limit_seconds": budget}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 200, 500])
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--densities", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--instance-seeds", type=int, nargs="+", default=[20261018, 20261019])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--definitions", choices=["weak", "strong"], nargs="+", default=["weak", "strong"])
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from .neural import solve_neural
    from .reference import solve_highs, solve_scip
    from .verify_report import verify
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # Deliberately warm imports/CUDA context and solver libraries outside measured trials.
    warm = HypergraphProblem(5, ((0, 1, 2), (2, 3)))
    solve_highs(warm, "weak", time_limit=1)
    solve_scip(warm, "weak", time_limit=1)
    solve_neural(warm, "weak", epochs=3, width=32, device=args.device)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "settings": settings, "git_commit": commit, "source_sha256": source_hashes,
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                                "scipy": importlib.metadata.version("scipy"), "dhg": importlib.metadata.version("dhg"),
                                "pyscipopt": importlib.metadata.version("pyscipopt"),
                                "torch_cpu_threads": torch.get_num_threads()},
                "timing": "2-second target by default; setup+training neural budget, final decoding extra; HiGHS solver budget; random restart stops between completed trials. Actual wall times saved. No aggregate best-of-three versus single-run claims.",
                "cases": []}
    existing_manifest = args.output / "manifest.json"
    if existing_manifest.exists():
        old = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if old["settings"] != settings or old["source_sha256"] != source_hashes:
            raise ValueError("Cannot resume a suite with different settings or source code")
        manifest = old
    else:
        write_json(existing_manifest, manifest)
    for n in args.sizes:
        for k in args.orders:
            for density in args.densities:
                for instance_seed in args.instance_seeds:
                    name = f"n{n}_k{k}_d{density}_s{instance_seed}"
                    instance = generate_instance(n, k, density, instance_seed)
                    problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
                    instance_path = args.output / "instances" / (name + ".json")
                    write_json(instance_path, instance)
                    for mode in args.definitions:
                        case_path = args.output / "cases" / (name + "_" + mode + ".json")
                        if case_path.exists():
                            verify(case_path)
                            print(f"RESUME {name} {mode}", flush=True)
                            continue
                        records = []
                        start = time.perf_counter()
                        selected = degree_greedy(problem, mode)
                        records.append({"method": "degree_greedy", "selected": selected, "size": len(selected),
                                        "feasible": problem.feasible(selected, mode), "total_seconds": time.perf_counter()-start})
                        records.append(solve_highs(problem, mode, args.budget))
                        records.append(solve_scip(problem, mode, args.budget))
                        for seed in args.model_seeds:
                            records.append(random_decoder(problem, mode, seed))
                            records.append(random_decoder(problem, mode, seed, args.budget))
                            records.append(solve_neural(problem, mode, seed=seed, epochs=10000,
                                                        device=args.device, time_limit=args.budget))
                        report = {"purpose": "synthetic benchmark round 1; fixed configuration without outcome-based selection",
                                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "definition": mode,
                                  "instance": {"name": name, "n": problem.n, "edges": problem.edges,
                                               "sha256": hashlib.sha256(instance_path.read_bytes()).hexdigest(),
                                               "generator": instance["generator"]},
                                  "git_commit": commit, "source_sha256": source_hashes,
                                  "environment": manifest["environment"], "records": records}
                        write_json(case_path, report)
                        verify(case_path)
                        if case_path.name not in manifest["cases"]:
                            manifest["cases"].append(case_path.name)
                        write_json(existing_manifest, manifest)
                        nn_sizes = [r["size"] for r in records if r["method"] == "hgnnp_pubo_repair"]
                        print(f"DONE {name} {mode}: nn={nn_sizes} greedy={records[0]['size']} "
                              f"milp={records[1]['size']} proven={records[1]['optimality_proven']}", flush=True)


if __name__ == "__main__":
    main()
