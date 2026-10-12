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
            "mip_gap": None if result.mip_gap is None else float(result.mip_gap),
            "time_limit_seconds": time_limit}
