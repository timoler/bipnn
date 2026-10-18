"""Incidence-count implementation of the existing deterministic MIS decoder.

The threshold, repair priority and extension order are unchanged. This module
does not replace the decoder in any already-frozen experiment protocol.
"""
import heapq
import math

from .problem import check_definition


def fast_decode_stages(problem, scores, mode):
    """Return exactly the stages/schema of controlled.decode_stages.

    During repair, selected vertices only disappear and violated edges can only
    become satisfied. Thus every future repair candidate was already in an
    initially violated edge; a fixed min-heap with lazy removal implements the
    original global minimum by (score, vertex). Edge counts and incidence lists
    then avoid repeatedly scanning the entire hypergraph.

    Index construction is included in each call. No state is cached between
    calls, and singleton edges/isolates follow the existing weak/strong rules.
    """
    check_definition(mode)
    if len(scores) != problem.n or any(not math.isfinite(p) for p in scores):
        raise ValueError("One finite score is required per vertex")
    n = problem.n
    edges = problem.edges
    incidence = [[] for _ in range(n)]
    selected = [score >= 0.5 for score in scores]
    raw = [v for v in range(n) if selected[v]]
    limits = [len(edge) - 1 if mode == "weak" else 1 for edge in edges]
    counts = []
    bad_incidence = [0] * n
    for edge_index, edge in enumerate(edges):
        count = sum(selected[v] for v in edge)
        counts.append(count)
        bad = count > limits[edge_index]
        for v in edge:
            incidence[v].append(edge_index)
            if bad:
                bad_incidence[v] += 1
    raw_feasible = not any(bad_incidence)
    candidates = [(scores[v], v) for v in raw if bad_incidence[v]]
    heapq.heapify(candidates)
    while candidates:
        _, vertex = heapq.heappop(candidates)
        if not bad_incidence[vertex]:
            continue
        selected[vertex] = False
        for edge_index in incidence[vertex]:
            was_bad = counts[edge_index] > limits[edge_index]
            counts[edge_index] -= 1
            if was_bad and counts[edge_index] <= limits[edge_index]:
                for v in edges[edge_index]:
                    bad_incidence[v] -= 1
    repaired = [v for v in range(n) if selected[v]]
    for vertex in sorted(range(n), key=lambda v: (-scores[v], v)):
        if selected[vertex]:
            continue
        if all(counts[i] + 1 <= limits[i] for i in incidence[vertex]):
            selected[vertex] = True
            for edge_index in incidence[vertex]:
                counts[edge_index] += 1
    final = [v for v in range(n) if selected[v]]
    return {
        "raw_selected": raw, "raw_size": len(raw), "raw_feasible": raw_feasible,
        "repaired_selected": repaired, "repaired_size": len(repaired),
        "selected": final, "size": len(final), "feasible": True,
        "removed_by_repair": len(raw) - len(repaired),
        "added_by_extension": len(final) - len(repaired),
    }


def fast_repair_and_extend(problem, scores, definition):
    """Drop-in solution-only equivalent of problem.repair_and_extend."""
    return fast_decode_stages(problem, scores, definition)["selected"]
