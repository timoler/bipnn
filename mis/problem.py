"""Unweighted MIS definitions and reference algorithms; no ML dependencies."""
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path


def check_definition(definition):
    if definition not in ("weak", "strong"):
        raise ValueError("definition must explicitly be 'weak' or 'strong'")


@dataclass(frozen=True)
class HypergraphProblem:
    n: int
    edges: tuple

    def __post_init__(self):
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive integer, including isolated vertices")
        cleaned = set()
        for edge in self.edges:
            if not edge:
                raise ValueError("Empty hyperedges are not supported")
            if any(type(v) is not int or not 0 <= v < self.n for v in edge):
                raise ValueError("Every vertex must be an integer in [0,n)")
            if len(set(edge)) != len(edge):
                raise ValueError("Repeated vertices within one edge are invalid")
            cleaned.add(tuple(sorted(edge)))
        object.__setattr__(self, "edges", tuple(sorted(cleaned)))

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(data["n"], tuple(tuple(e) for e in data["edges"]))

    def violations(self, selected, definition):
        check_definition(definition)
        chosen = set(selected)
        if any(type(v) is not int or not 0 <= v < self.n for v in chosen):
            raise ValueError("Invalid selected vertex")
        return [e for e in self.edges if
                (all(v in chosen for v in e) if definition == "weak"
                 else sum(v in chosen for v in e) > 1)]

    def feasible(self, selected, definition):
        return not self.violations(selected, definition)

    def terms(self, definition, penalty=2.0):
        """Polynomial dictionary. For unweighted MIS, penalty > 1 is sufficient.

        Every infeasible binary assignment contains a violating edge/pair.
        Removing one selected vertex increases -|S| by 1 and removes at least
        one penalty > 1, so an infeasible point cannot be a global minimum.
        This does not guarantee gradient descent reaches the global minimum.
        """
        check_definition(definition)
        if not math.isfinite(penalty) or penalty <= 1:
            raise ValueError("Use a finite penalty > 1 for the unweighted guarantee")
        terms = {(i,): -1.0 for i in range(self.n)}
        penalties = self.edges if definition == "weak" else sorted({
            pair for e in self.edges for pair in combinations(e, 2)})
        for edge in penalties:
            terms[edge] = terms.get(edge, 0.0) + penalty
        return terms


def exact_mis(problem, definition, max_nodes=22):
    """Exhaustive reference solver, deliberately restricted to small instances."""
    check_definition(definition)
    if problem.n > max_nodes:
        raise ValueError("Exact enumeration is limited to small verification instances")
    for size in range(problem.n, -1, -1):
        for selected in combinations(range(problem.n), size):
            if problem.feasible(selected, definition):
                return list(selected)
    raise AssertionError("The empty set must be feasible for supported instances")


def repair_and_extend(problem, scores, definition):
    """Repair a thresholded set, then greedily extend it; return a feasible set.

    This is explicit postprocessing, not a neural feasibility guarantee.
    """
    check_definition(definition)
    if len(scores) != problem.n or any(not math.isfinite(p) for p in scores):
        raise ValueError("Provide one finite score per vertex")
    selected = {i for i, p in enumerate(scores) if p >= 0.5}
    while True:
        bad = problem.violations(selected, definition)
        if not bad:
            break
        candidates = {v for e in bad for v in e if v in selected}
        selected.remove(min(candidates, key=lambda v: (scores[v], v)))
    for v in sorted(range(problem.n), key=lambda v: (-scores[v], v)):
        if v not in selected and problem.feasible(selected | {v}, definition):
            selected.add(v)
    return sorted(selected)


def degree_greedy(problem, definition):
    check_definition(definition)
    degrees = [sum(v in e for e in problem.edges) for v in range(problem.n)]
    selected = set()
    for v in sorted(range(problem.n), key=lambda v: (degrees[v], v)):
        if problem.feasible(selected | {v}, definition):
            selected.add(v)
    return sorted(selected)
