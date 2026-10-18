"""Fast implementations for new protocols; frozen baselines remain unchanged.

Only bookkeeping is accelerated: the degree order and random-score decoding
rules match the original implementations. SCIP receives the same degree-greedy
solution, with a total-time target that includes warm-start/model construction.
"""
import math
import random
import time

from .fast_decode import fast_decode_stages
from .problem import check_definition


def fast_degree_greedy(problem, definition):
    """Return exactly ``problem.degree_greedy`` using incidence counts.

    The degree is the number of original hyperedges containing each vertex;
    it is not a clique-expansion degree. The static order is (degree, vertex).
    Counts maintain the same acceptance rule as checking every edge after
    each attempted insertion, including weak singleton constraints and isolates.
    """
    check_definition(definition)
    incidence = [[] for _ in range(problem.n)]
    limits = []
    for edge_index, edge in enumerate(problem.edges):
        limits.append(len(edge) - 1 if definition == "weak" else 1)
        for vertex in edge:
            incidence[vertex].append(edge_index)
    counts = [0] * len(problem.edges)
    selected = []
    for vertex in sorted(range(problem.n), key=lambda v: (len(incidence[v]), v)):
        if all(counts[i] < limits[i] for i in incidence[vertex]):
            selected.append(vertex)
            for edge_index in incidence[vertex]:
                counts[edge_index] += 1
    return sorted(selected)


def random_fast(problem, definition, seed, budget=None):
    """Decode one random score vector or restart until a complete trial ends.

    The best complete trial is retained, including its actual scores and all
    decoder stages. Ties retain the earliest trial, as in the frozen baseline.
    The target includes score generation, index construction and decoding;
    checking it only between trials can cause a small overrun.
    """
    check_definition(definition)
    if budget is not None and (not math.isfinite(budget) or budget <= 0):
        raise ValueError("budget must be finite and positive, or None")
    rng = random.Random(seed)
    start = time.perf_counter()
    best = None
    trials = 0
    while True:
        scores = [rng.random() for _ in range(problem.n)]
        decoded = fast_decode_stages(problem, scores, definition)
        trials += 1
        if best is None or decoded["size"] > best["size"]:
            best = {**decoded, "scores": scores}
        if budget is None or time.perf_counter() - start >= budget:
            break
    return {
        **best,
        "method": "random_score_same_decoder" if budget is None else "random_restart_same_decoder",
        "implementation": "incidence_counts_same_decoder",
        "seed": seed,
        "trials": trials,
        "total_seconds": time.perf_counter() - start,
        "time_limit_seconds": budget,
        "budget_policy": "check total elapsed time after each complete decoding trial",
    }


def _json_scip_number(value, model):
    """Preserve finite solver values while disclosing SCIP infinity sentinels."""
    value = float(value)
    if math.isnan(value):
        return None, "nan", False, None
    infinite = math.isinf(value)
    if infinite or model.isInfinity(abs(value)):
        return (None, "ieee_infinity" if infinite else "scip_infinity", True,
                "positive" if value > 0 else "negative")
    return value, "finite", False, None


def _cutoff_snapshot(incumbents, bounds, terminal, deadline, completion_seconds):
    """Select only fully materialized evidence available by a wall deadline."""
    eligible = [event for event in incumbents if event["available_seconds"] <= deadline]
    if not eligible:
        raise ValueError("The time-zero empty incumbent must always be recorded")
    best = max(eligible, key=lambda event: event["size"])
    eligible_bounds = [event for event in bounds if event["available_seconds"] <= deadline]
    bound = eligible_bounds[-1]
    completed = terminal["available_seconds"] <= deadline
    upper = bound["upper_bound"]
    size = best["size"]
    if upper < size - 1e-6:
        raise RuntimeError("Cutoff bound contradicts a cutoff feasible incumbent")
    gap = ((upper - size) / size if size else 0. if upper == 0 else None)
    return {
        "selected": list(best["selected"]), "size": size, "feasible": True,
        "available_seconds": best["available_seconds"], "source": best["source"],
        "upper_bound": upper, "bound_kind": bound["bound_kind"],
        "upper_bound_available_seconds": bound["available_seconds"],
        "raw_upper_bound": bound.get("raw_upper_bound"),
        "raw_upper_bound_flag": bound.get("raw_upper_bound_flag", "not_observed"),
        "mip_gap": gap,
        "mip_gap_kind": "derived_from_cutoff_incumbent_and_recorded_upper_bound",
        "mip_gap_flag": "finite" if gap is not None else "undefined_zero_incumbent",
        "optimality_proven": bool(completed and terminal["optimality_proven"]),
        "status": terminal["status"] if completed else "deadline_snapshot",
        "deadline_seconds": deadline, "time_limit_seconds": deadline,
        "total_seconds": min(completion_seconds, deadline),
        "completion_total_seconds": completion_seconds,
        "solver_completed_before_deadline": completed,
        "overshoot": completion_seconds > deadline,
        "overshoot_seconds": max(0., completion_seconds - deadline),
        "deadline_missed": completion_seconds > deadline,
        "policy": "first largest complete incumbent and latest bound available by deadline; terminal evidence requires cleanup completion",
    }


def _scip_bound_fields(model, n):
    raw, flag, infinite, sign = _json_scip_number(model.getDualbound(), model)
    upper = float(n) if raw is None else min(float(n), raw)
    kind = ("trivial_node_count_bound" if raw is None else
            "scip_dual_bound_capped_by_node_count" if raw > n else "scip_dual_bound")
    return {
        "upper_bound": upper, "bound_kind": kind, "raw_upper_bound": raw,
        "raw_upper_bound_flag": flag, "raw_upper_bound_is_infinite": infinite,
        "raw_upper_bound_infinity_sign": sign,
    }


def solve_scip_fast(problem, definition, time_limit=5.0, *, cutoff_events=False):
    """Direct MILP with an equivalent fast greedy warm start and total target.

    Imports are outside the measured region, matching library warm-up in the
    experiment runners. Warm-start/model construction is deducted from the
    solver limit, with a minimum 1 ms solve so the feasible incumbent survives
    even when construction consumes the target. Actual elapsed time includes
    extraction, independent feasibility checking and solver cleanup. A budget
    target is therefore not a strict wall-clock deadline. With ``cutoff_events``
    enabled, a separate cutoff snapshot uses only fully extracted incumbent
    events available by the target; final fields still describe the complete run.
    """
    from pyscipopt import Model, quicksum, Eventhdlr, SCIP_EVENTTYPE

    check_definition(definition)
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    start = time.perf_counter()
    incumbent_events = [{"source": "initial_empty", "selected": [], "size": 0,
                         "feasible": True, "available_seconds": 0.}]
    bound_events = [{"source": "trivial_node_count", "upper_bound": float(problem.n),
                     "bound_kind": "trivial_node_count_bound", "available_seconds": 0.,
                     "raw_upper_bound": None, "raw_upper_bound_flag": "not_observed"}]
    initial = set(fast_degree_greedy(problem, definition))
    model = Model("hypergraph_mis_fast_warm_start")
    try:
        model.hideOutput()
        model.setIntParam("parallel/maxnthreads", 1)
        model.setIntParam("randomization/randomseedshift", 0)
        x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(problem.n)]
        for edge in problem.edges:
            upper = len(edge) - 1 if definition == "weak" else 1
            model.addCons(quicksum(x[i] for i in edge) <= upper)
        model.setObjective(quicksum(x), "maximize")
        warm = model.createSol()
        for vertex in range(problem.n):
            model.setSolVal(warm, x[vertex], float(vertex in initial))
        if not model.addSol(warm, free=True):
            raise RuntimeError("SCIP rejected a verified feasible degree-greedy warm start")
        if cutoff_events:
            warm_event = {"source": "degree_greedy_warm_start", "selected": sorted(initial),
                          "size": len(initial), "feasible": True}
            warm_event["available_seconds"] = time.perf_counter() - start
            incumbent_events.append(warm_event)

            class IncumbentTimeline(Eventhdlr):
                """Callbacks are timed after extracting and checking full solutions."""
                callback_count = 0
                callback_seconds = 0.

                def __init__(self):
                    self.errors = []

                def eventinit(self):
                    self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
                    self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)
                    self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

                def eventexit(self):
                    self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
                    self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)
                    self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)

                def eventexec(self, event):
                    callback_start = time.perf_counter()
                    self.callback_count += 1
                    try:
                        if event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND:
                            solution = self.model.getBestSol()
                            if solution is not None:
                                chosen = [i for i in range(problem.n)
                                          if self.model.getSolVal(solution, x[i]) >= .5]
                                if not problem.feasible(chosen, definition):
                                    raise RuntimeError("SCIP event incumbent failed independent feasibility check")
                                if len(chosen) > incumbent_events[-1]["size"]:
                                    record = {"source": "scip_best_solution_event", "selected": chosen,
                                              "size": len(chosen), "feasible": True}
                                    record["available_seconds"] = time.perf_counter() - start
                                    incumbent_events.append(record)
                        bound = _scip_bound_fields(self.model, problem.n)
                        previous = bound_events[-1]
                        if (bound["upper_bound"] != previous["upper_bound"] or
                                bound["raw_upper_bound_flag"] != previous["raw_upper_bound_flag"]):
                            record = {**bound, "source": "scip_progress_event"}
                            record["available_seconds"] = time.perf_counter() - start
                            bound_events.append(record)
                    except Exception as error:
                        # Exceptions escaping a C callback may be printed and ignored.
                        # Refuse the run explicitly after optimize instead.
                        self.errors.append(repr(error))
                    finally:
                        self.callback_seconds += time.perf_counter() - callback_start

            handler = IncumbentTimeline()
            model.includeEventhdlr(handler, "incumbent_wall_timeline",
                                   "Complete incumbents and bounds with wall-clock availability")
        build_seconds = time.perf_counter() - start
        solver_budget = max(0.001, time_limit - build_seconds)
        model.setRealParam("limits/time", solver_budget)
        model.optimize()
        if cutoff_events and handler.errors:
            raise RuntimeError("SCIP timeline callback failed: " + "; ".join(handler.errors))
        solution = model.getBestSol()
        if solution is None:
            raise RuntimeError("SCIP returned no incumbent despite its feasible warm start")
        selected = [i for i in range(problem.n) if model.getSolVal(solution, x[i]) >= 0.5]
        if not problem.feasible(selected, definition):
            raise RuntimeError("SCIP incumbent failed independent feasibility verification")
        if len(selected) < len(initial):
            raise RuntimeError("SCIP incumbent lost the feasible warm start")
        raw_bound, bound_flag, bound_infinite, bound_sign = _json_scip_number(model.getDualbound(), model)
        gap, gap_flag, gap_infinite, gap_sign = _json_scip_number(model.getGap(), model)
        if gap is not None and gap < 0:
            gap, gap_flag = None, "undefined_negative"
        if raw_bound is None:
            upper_bound = float(problem.n)
            bound_kind = "trivial_node_count_bound"
        else:
            if raw_bound < len(selected) - 1e-6:
                raise RuntimeError("SCIP dual upper bound contradicts its feasible incumbent")
            upper_bound = min(float(problem.n), raw_bound)
            bound_kind = ("scip_dual_bound_capped_by_node_count"
                          if raw_bound > problem.n else "scip_dual_bound")
        status = str(model.getStatus())
        result = {
            "method": "scip_milp_greedy_start",
            "implementation": "incidence_counts_degree_warm_start_total_budget",
            "selected": selected, "size": len(selected), "feasible": True,
            "status": status, "optimality_proven": status == "optimal",
            "upper_bound": upper_bound, "bound_kind": bound_kind,
            "raw_upper_bound": raw_bound,
            "raw_upper_bound_flag": bound_flag,
            "raw_upper_bound_is_infinite": bound_infinite,
            "raw_upper_bound_infinity_sign": bound_sign,
            "mip_gap": gap, "mip_gap_flag": gap_flag,
            "mip_gap_is_infinite": gap_infinite,
            "mip_gap_infinity_sign": gap_sign,
            "time_limit_seconds": time_limit,
            "solver_time_limit_seconds": solver_budget,
            "build_seconds": build_seconds,
            "solver_seconds": float(model.getSolvingTime()),
            "warm_start": "degree_greedy",
            "warm_start_implementation": "incidence_counts_exact_same_order",
            "warm_start_size": len(initial), "threads": 1,
            "random_seed_shift": 0,
            "budget_policy": "total target minus Python model and warm-start construction; minimum 1ms solver; cleanup included in actual total",
        }
        if cutoff_events:
            result["terminal_observed_seconds"] = time.perf_counter() - start
            result["event_callback_count"] = handler.callback_count
            result["event_callback_seconds"] = handler.callback_seconds
    finally:
        model.freeProb()
    result["total_seconds"] = time.perf_counter() - start
    if cutoff_events:
        terminal_seconds = result["total_seconds"]
        terminal = {"source": "terminal_after_cleanup", "selected": list(selected),
                    "size": len(selected), "feasible": True, "status": status,
                    "optimality_proven": result["optimality_proven"],
                    "available_seconds": terminal_seconds}
        incumbent_events.append(terminal)
        bound_events.append({key: result[key] for key in (
            "upper_bound", "bound_kind", "raw_upper_bound", "raw_upper_bound_flag",
            "raw_upper_bound_is_infinite", "raw_upper_bound_infinity_sign")})
        bound_events[-1].update({"source": "terminal_after_cleanup",
                                 "available_seconds": terminal_seconds})
        result.update({"incumbent_events": incumbent_events, "bound_events": bound_events,
                       "terminal_available_seconds": terminal_seconds,
                       "completion_available_seconds": terminal_seconds,
                       "cutoff": _cutoff_snapshot(incumbent_events, bound_events, terminal,
                                                   time_limit, terminal_seconds)})
        result["report_assembly_seconds"] = time.perf_counter() - start - terminal_seconds
    return result
