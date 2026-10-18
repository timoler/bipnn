"""Direct binary linear model solved with SciPy's bundled HiGHS solver."""
import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .problem import check_definition


def solve_highs(problem, definition, time_limit=30.0):
    check_definition(definition)
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    start = time.perf_counter()
    rows, columns = [], []
    for j, edge in enumerate(problem.edges):
        rows.extend([j] * len(edge))
        columns.extend(edge)
    constraints = None
    if problem.edges:
        a = coo_matrix((np.ones(len(rows)), (rows, columns)),
                       shape=(len(problem.edges), problem.n)).tocsc()
        upper = np.array([len(e)-1 if definition == "weak" else 1 for e in problem.edges])
        constraints = LinearConstraint(a, -np.inf, upper)
    result = milp(-np.ones(problem.n), integrality=np.ones(problem.n),
                  bounds=Bounds(np.zeros(problem.n), np.ones(problem.n)),
                  constraints=constraints, options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if result.x is None:
        raise RuntimeError(f"HiGHS returned no incumbent: {result.message}")
    selected = np.flatnonzero(result.x >= 0.5).tolist()
    if not problem.feasible(selected, definition):
        raise RuntimeError("HiGHS incumbent failed independent feasibility verification")
    elapsed = time.perf_counter() - start
    return {"method": "highs_milp", "selected": selected, "size": len(selected),
            "feasible": True, "total_seconds": elapsed, "status": int(result.status),
            "optimality_proven": result.status == 0,
            "upper_bound": None if result.mip_dual_bound is None else -float(result.mip_dual_bound),
            "mip_gap": None if result.mip_gap is None else float(result.mip_gap),
            "time_limit_seconds": time_limit}


def solve_scip(problem, definition, time_limit=2.0, *, total_budget=False):
    """Direct MILP with an explicitly disclosed degree-greedy warm start."""
    from pyscipopt import Model, quicksum
    from .problem import degree_greedy
    check_definition(definition)
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    start = time.perf_counter()
    initial = set(degree_greedy(problem, definition))
    model = Model("hypergraph_mis")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", 0)
    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(problem.n)]
    for edge in problem.edges:
        model.addCons(quicksum(x[i] for i in edge) <= (len(edge)-1 if definition == "weak" else 1))
    model.setObjective(quicksum(x), "maximize")
    warm = model.createSol()
    for i in range(problem.n):
        model.setSolVal(warm, x[i], float(i in initial))
    if not model.addSol(warm, free=True):
        raise RuntimeError("SCIP rejected a verified feasible warm start")
    construction_seconds = time.perf_counter() - start
    solver_budget = max(0.001, time_limit - construction_seconds) if total_budget else time_limit
    model.setRealParam("limits/time", solver_budget)
    model.optimize()
    selected = [i for i in range(problem.n) if model.getVal(x[i]) >= 0.5]
    if not problem.feasible(selected, definition):
        raise RuntimeError("SCIP incumbent failed independent feasibility verification")
    result = {"method": "scip_milp_greedy_start", "selected": selected, "size": len(selected),
              "feasible": True, "total_seconds": time.perf_counter()-start,
              "status": str(model.getStatus()), "optimality_proven": str(model.getStatus()) == "optimal",
              "upper_bound": min(float(problem.n), float(model.getDualbound())),
              "mip_gap": float(model.getGap()), "time_limit_seconds": time_limit,
              "solver_seconds": float(model.getSolvingTime()),
              "warm_start": "degree_greedy", "warm_start_size": len(initial), "threads": 1}
    if total_budget:
        result.update({"budget_policy": "total target minus Python model and warm-start construction; minimum 1ms solver",
                       "build_seconds": construction_seconds, "solver_time_limit_seconds": solver_budget})
    model.freeProb()
    return result
