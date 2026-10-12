"""Run an auditable small-instance experiment; no claim of benchmark superiority."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .problem import HypergraphProblem, degree_greedy, exact_mis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--definition", choices=["weak", "strong"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--milp-time-limit", type=float, default=30.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .neural import solve_neural
    from .reference import solve_highs
    import torch

    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)

    problem = HypergraphProblem.from_json(args.instance)
    records = []
    solvers = [("degree_greedy", degree_greedy)]
    if problem.n <= 22:
        solvers.append(("exact_enumeration", exact_mis))
    for name, solver in solvers:
        start = time.perf_counter()
        selected = solver(problem, args.definition)
        records.append({"method": name, "selected": selected, "size": len(selected),
                        "feasible": problem.feasible(selected, args.definition),
                        "total_seconds": time.perf_counter() - start})
    records.append(solve_highs(problem, args.definition, args.milp_time_limit))
    for seed in args.seeds:
        result = solve_neural(problem, args.definition, seed=seed,
                              epochs=args.epochs, device=args.device)
        records.append(result)
        print(f"{args.definition} seed={seed}: size={result['size']} "
              f"raw_feasible={result['raw_feasible']} seconds={result['total_seconds']:.3f}", flush=True)
    root = Path(__file__).resolve().parents[1]
    def git(*parameters):
        try:
            return subprocess.check_output(["git", "-C", str(root), *parameters], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    report = {
        "purpose": "small-instance pipeline verification, not a formal benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": args.definition,
        "instance": {"name": args.instance.name, "n": problem.n, "edges": problem.edges,
                     "sha256": hashlib.sha256(args.instance.read_bytes()).hexdigest()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda,
                        "dhg": importlib.metadata.version("dhg"), "cpu_threads": args.threads,
                        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
