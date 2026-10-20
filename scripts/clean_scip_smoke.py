"""Save exact-checkable SCIP results from the clean environment; no benchmark claim."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mis.fast_baselines import solve_scip_fast
from mis.problem import HypergraphProblem, exact_mis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    instance = ROOT / "instances/toy_overlap.json"
    problem = HypergraphProblem.from_json(instance)
    for mode in ("weak", "strong"):
        result = solve_scip_fast(problem, mode, time_limit=2.0, cutoff_events=True)
        optimum = len(exact_mis(problem, mode))
        assert result["optimality_proven"] and result["size"] == optimum
        assert result["cutoff"]["optimality_proven"] and result["cutoff"]["size"] == optimum
        report = {
            "purpose": "clean-environment SCIP pipeline verification, not a performance comparison",
            "definition": mode,
            "instance": {"name": instance.name, "n": problem.n, "edges": problem.edges,
                         "sha256": hashlib.sha256(instance.read_bytes()).hexdigest()},
            "pyscipopt_version": importlib.metadata.version("pyscipopt"),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ["mis/fast_baselines.py", "mis/fast_decode.py", "mis/problem.py"]},
            "records": [result],
        }
        path = args.output / ("smoke-scip-" + mode + ".json")
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"definition": mode, "exact_size": optimum,
                          "scip_size": result["size"], "optimality_proven": result["optimality_proven"]}))


if __name__ == "__main__":
    main()
