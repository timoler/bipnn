"""Independently verify saved solutions against the saved instance."""
import argparse
import json
import math
from pathlib import Path

from .problem import HypergraphProblem, exact_mis


def verify(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    instance = data["instance"]
    problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
    mode = data["definition"]
    optimum = len(exact_mis(problem, mode)) if problem.n <= 22 else None
    for record in data["records"]:
        selected = record["selected"]
        if len(set(selected)) != len(selected) or len(selected) != record["size"]:
            raise ValueError("Recorded solution size mismatch")
        if not record["feasible"] or not problem.feasible(selected, mode):
            raise ValueError("Recorded solution is infeasible")
        if not math.isfinite(record["total_seconds"]) or record["total_seconds"] < 0:
            raise ValueError("Invalid runtime")
        if optimum is not None and record["size"] > optimum:
            raise ValueError("Recorded size exceeds exact optimum")
        if record["method"] == "exact_enumeration" and record["size"] != optimum:
            raise ValueError("Incorrect exact reference")
        if "raw_selected" in record:
            if len(record["raw_selected"]) != record["raw_size"]:
                raise ValueError("Raw size mismatch")
            if problem.feasible(record["raw_selected"], mode) != record["raw_feasible"]:
                raise ValueError("Raw feasibility mismatch")
            thresholded = [i for i, p in enumerate(record["scores"]) if p >= 0.5]
            if thresholded != record["raw_selected"]:
                raise ValueError("Raw solution does not match saved probabilities")
    return {"report": Path(path).name, "definition": mode, "exact_size": optimum,
            "verified_records": len(data["records"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    for report in args.reports:
        print(json.dumps(verify(report)))
