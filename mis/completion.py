"""Separate completion experiments; all earlier algorithms/protocols stay frozen.

The public network/training step and the same indexed MIS objective are reused.
Fixed epochs control optimization steps; timed controls use explicit horizons.
Neither experiment silently changes decoding to a reduced constraint system.
"""
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from dhg import Hypergraph

from .fast_decode import fast_decode_stages
from .indexed_loss import indexed_tensors, indexed_polynomial_loss
from .problem import HypergraphProblem, check_definition
from src.models import Layer, LayerType, Net
from src.train import train


def _tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_digest(net):
    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf8"))
        digest.update(b"\x00")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate(problem, mode, annealing, penalty, width, device):
    check_definition(mode)
    if annealing not in ("cra", "none"):
        raise ValueError("annealing must be 'cra' or 'none'")
    if not math.isfinite(penalty) or penalty <= 1:
        raise ValueError("Use a finite penalty > 1")
    if type(width) is not int or width < 2 or width % 2:
        raise ValueError("width must be a positive even integer >= 2")
    if problem.n < 2:
        raise ValueError("Training-mode public BatchNorm requires at least two vertices")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _build(problem, mode, width, penalty, device, objective_problem, message_problem):
    _, indices, coefficients = indexed_tensors(objective_problem, mode, penalty, device)
    # Only monomial membership is passed to Hypergraph, never its signed coefficient.
    message_edges = list(message_problem.terms(mode, penalty))
    graph = Hypergraph(problem.n, message_edges, device=device)
    net = Net([
        Layer(LayerType.HGNNPCONV, width, width),
        Layer(LayerType.HGNNPCONV, width, width // 2, last_conv=True),
        Layer(LayerType.LINEAR, width // 2, 1, dropout=0.),
    ]).to(device)
    features = torch.rand(problem.n, width, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-8)
    state_hash = _state_digest(net)
    feature_hash = _tensor_digest(features)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    structure = {
        "objective_constraint_count": len(objective_problem.edges),
        "objective_monomial_count": int(indices.shape[0]),
        "objective_max_order": int(indices.shape[1]),
        "message_constraint_count": len(message_problem.edges),
        "message_monomial_count": len(message_edges),
        "decode_constraint_count": len(problem.edges),
        "initial_state_sha256": state_hash, "feature_sha256": feature_hash,
    }
    return net, features, graph, optimizer, indices, coefficients, structure


def _check_gradients(net):
    finite = [torch.isfinite(parameter.grad).all()
              for parameter in net.parameters() if parameter.grad is not None]
    if not finite or not bool(torch.stack(finite).all().item()):
        raise FloatingPointError("Missing or nonfinite completion-experiment gradients")


def _training_step(net, features, graph, optimizer, indices, coefficients, gamma):
    def objective(outs, **kwargs):
        energy = indexed_polynomial_loss(outs, indices, coefficients)
        return energy + gamma * (1 - (2 * outs - 1).square()).sum()

    loss, outs = train(net, features, graph, optimizer, objective)
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite completion-experiment loss")
    return loss, outs


def _collect(problem, mode, outs, epoch, loss, gamma, role, start):
    begin = time.perf_counter()
    scores = outs.detach().flatten().cpu().tolist()
    decoded = fast_decode_stages(problem, scores, mode)
    candidate = {**decoded, "scores": scores, "epoch": epoch, "role": role,
                 "loss_before_update": loss, "gamma": gamma,
                 "annealing_coefficient": gamma}
    candidate["available_seconds"] = time.perf_counter() - start
    return candidate, time.perf_counter() - begin


def _base_settings(width, penalty, annealing, structure):
    return {
        "width": width, "hidden_widths": [width, width // 2],
        "features": "fixed per-node uniform U[0,1)",
        "annealing": annealing, "penalty": penalty,
        "lr": 1e-4, "weight_decay": 5e-8, "effective_dropout": 0.,
        "output": "training-mode forward before parameter update of that step",
        "gradient_check": "finite loss each step; final completed gradient check only",
        "decode_problem": "original problem, even if objective/message constraints differ",
        **structure,
    }


def solve_fixed_epochs(problem, mode, seed, *, epochs=1600, width=1024,
                       penalty=2., annealing="cra", objective_problem=None,
                       message_problem=None, checkpoint_every=100, device="cuda"):
    """Optimize for identical step counts; elapsed times are descriptive only.

    The first candidate is the untrained pre-update output returned after step
    one completes. Further candidates use exactly the requested step multiples
    and the final step, without duplicating step one. No empty incumbent is
    inserted into a fixed-step experiment.
    """
    device = _validate(problem, mode, annealing, penalty, width, device)
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    objective_problem = problem if objective_problem is None else objective_problem
    message_problem = problem if message_problem is None else message_problem
    if objective_problem.n != problem.n or message_problem.n != problem.n:
        raise ValueError("Objective, message and decoding problems need the same explicit n")
    _seed(seed, device)
    start = time.perf_counter()
    net, features, graph, optimizer, indices, coefficients, structure = _build(
        problem, mode, width, penalty, device, objective_problem, message_problem)
    built = time.perf_counter()
    candidates = []
    training_seconds, decode_seconds = 0., 0.
    for epoch in range(1, epochs + 1):
        progress = (epoch - 1) / (epochs - 1) if epochs > 1 else 0.
        gamma = -1 + 1.6 * progress if annealing == "cra" else 0.
        begin = time.perf_counter()
        loss, outs = _training_step(net, features, graph, optimizer, indices, coefficients, gamma)
        training_seconds += time.perf_counter() - begin
        if epoch == 1 or epoch % checkpoint_every == 0 or epoch == epochs:
            role = ("untrained_first_forward" if epoch == 1 else
                    "final_epoch" if epoch == epochs else "epoch_checkpoint")
            candidate, elapsed = _collect(problem, mode, outs, epoch, loss, gamma, role, start)
            candidate["coefficient_progress"] = progress
            candidates.append(candidate)
            decode_seconds += elapsed
    begin = time.perf_counter()
    _check_gradients(net)
    training_seconds += time.perf_counter() - begin
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total = time.perf_counter() - start
    best_index = max(range(len(candidates)), key=lambda index: candidates[index]["size"])
    latest_index = len(candidates) - 1
    return {
        "method": "bipnn_fixed_epochs_completion", "definition": mode, "seed": seed,
        "candidates": candidates, "initial_index": 0,
        "best_index": best_index, "latest_index": latest_index, "last_index": latest_index,
        "initial": candidates[0], "best": candidates[best_index], "latest": candidates[latest_index],
        "epochs_completed": epochs, "total_seconds": total, "run_total_seconds": total,
        "build_seconds": built - start, "training_seconds": training_seconds,
        "decode_seconds": decode_seconds, "online_decode_seconds": decode_seconds,
        "other_seconds": max(0., total - (built - start) - training_seconds - decode_seconds),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "initial_state_sha256": structure["initial_state_sha256"],
        "feature_sha256": structure["feature_sha256"],
        "settings": {**_base_settings(width, penalty, annealing, structure),
                     "fixed_epochs": epochs, "checkpoint_every": checkpoint_every,
                     "schedule": "-1+1.6*(epoch-1)/(epochs-1); first=-1 if epochs=1" if annealing == "cra" else "0",
                     "selection": "first maximum decoded candidate at the declared fixed step checkpoints",
                     "timing": "fixed optimization steps; wall-clock times are descriptive, not equal-budget comparisons"},
    }


def _checked_targets(targets):
    values = [float(value) for value in targets]
    if (not values or values != sorted(set(values)) or
            any(not math.isfinite(value) or value <= 0 for value in values)):
        raise ValueError("Targets must be finite, positive and strictly increasing")
    return values


def _empty_candidate():
    return {"selected": [], "size": 0, "feasible": True, "available_seconds": 0.,
            "role": "empty_initial_incumbent", "epoch": 0, "trial": 0}


def _prefix_snapshot(candidates, deadline, emitted):
    eligible = [i for i, candidate in enumerate(candidates) if candidate["available_seconds"] <= deadline]
    if not eligible:
        raise ValueError("Missing the time-zero feasible incumbent")
    best = max(eligible, key=lambda index: candidates[index]["size"])
    return {
        **candidates[best], "budget_seconds": deadline, "cutoff_seconds": deadline,
        "total_seconds": emitted, "snapshot_emitted_seconds": emitted,
        "overshoot_seconds": max(0., emitted - deadline), "deadline_missed": emitted > deadline,
        "best_candidate_index": best, "latest_candidate_index": eligible[-1],
        "candidate_count": len(eligible), "best_available_seconds": candidates[best]["available_seconds"],
    }


def solve_timed_control(problem, mode, seed, *, annealing, penalty=2.,
                        targets=(5, 15, 30), interval=1., width=1024, device="cuda"):
    """Explicit CRA/penalty controls with the frozen anytime sampling policy.

    One run's horizon is targets[-1]. Passing (5,) or (15,) performs an
    independent compressed schedule; (5,15,30) retains the 30-second schedule
    for all three snapshots. Future terminal outputs never enter a past prefix.
    """
    device = _validate(problem, mode, annealing, penalty, width, device)
    targets = _checked_targets(targets)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be finite and positive")
    _seed(seed, device)
    start = time.perf_counter()
    net, features, graph, optimizer, indices, coefficients, structure = _build(
        problem, mode, width, penalty, device, problem, problem)
    built = time.perf_counter()
    horizon = targets[-1]
    due = sorted({i * interval for i in range(1, math.ceil(horizon / interval))} |
                 {target - min(.1, target / 10) for target in targets})
    candidates, snapshots = [_empty_candidate()], []
    epochs, due_index = 0, 0
    training_seconds, decode_seconds = 0., 0.
    initial_index = None

    def collect(outs, role, loss, gamma, coefficient_elapsed):
        nonlocal decode_seconds, initial_index
        candidate, elapsed = _collect(problem, mode, outs, epochs, loss, gamma, role, start)
        candidate["coefficient_elapsed_seconds"] = coefficient_elapsed
        candidates.append(candidate)
        decode_seconds += elapsed
        if role == "untrained_first_forward":
            initial_index = len(candidates) - 1

    def emit_ready():
        while len(snapshots) < len(targets) and time.perf_counter() - start >= targets[len(snapshots)]:
            target = targets[len(snapshots)]
            record = _prefix_snapshot(candidates, target, time.perf_counter() - start)
            record.update({"method": "bipnn_completion_timed_control", "seed": seed,
                           "config": f"w{width}_{annealing}",
                           "epochs_completed_at_snapshot": epochs,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
            snapshots.append(record)

    while time.perf_counter() - start < horizon or epochs == 0:
        elapsed = time.perf_counter() - start
        gamma = -1 + 1.6 * min(1., elapsed / horizon) if annealing == "cra" else 0.
        begin = time.perf_counter()
        loss, outs = _training_step(net, features, graph, optimizer, indices, coefficients, gamma)
        training_seconds += time.perf_counter() - begin
        epochs += 1
        now = time.perf_counter() - start
        if epochs == 1:
            collect(outs, "untrained_first_forward", loss, gamma, elapsed)
        elif due_index < len(due) and now >= due[due_index]:
            collect(outs, "periodic_training_forward", loss, gamma, elapsed)
        while due_index < len(due) and due[due_index] <= time.perf_counter() - start:
            due_index += 1
        emit_ready()
    emit_ready()
    collect(outs, "terminal_diagnostic", loss, gamma, elapsed)
    emit_ready()
    begin = time.perf_counter()
    _check_gradients(net)
    training_seconds += time.perf_counter() - begin
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total = time.perf_counter() - start
    final_snapshot = snapshots[-1]
    best_index = final_snapshot["best_candidate_index"]
    latest_index = final_snapshot["latest_candidate_index"]
    return {
        "method": "bipnn_completion_timed_control", "definition": mode, "seed": seed,
        "config": f"w{width}_{annealing}", "records": snapshots,
        "candidates": candidates, "initial_index": initial_index,
        "last_index": len(candidates) - 1, "best_index": best_index, "latest_index": latest_index,
        "initial": candidates[initial_index], "best": candidates[best_index],
        "latest": candidates[latest_index], "terminal": candidates[-1],
        "epochs_completed": epochs, "total_seconds": total, "run_total_seconds": total,
        "build_seconds": built - start, "training_seconds": training_seconds,
        "decode_seconds": decode_seconds, "online_decode_seconds": decode_seconds,
        "other_seconds": max(0., total - (built - start) - training_seconds - decode_seconds),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "initial_state_sha256": structure["initial_state_sha256"],
        "feature_sha256": structure["feature_sha256"],
        "settings": {**_base_settings(width, penalty, annealing, structure),
                     "targets_seconds": targets, "horizon_seconds": horizon,
                     "candidate_interval_seconds": interval, "pre_cutoff_margin_seconds": .1,
                     "schedule": "-1+1.6*min(1,elapsed/horizon)" if annealing == "cra" else "0",
                     "selection": "first maximum feasible candidate completed by cutoff",
                     "timing": "construction, hashes, training and online decoding included; actual emission latency saved"},
    }


def prune_weak_constraints(problem):
    """Drop Boolean weak constraints implied by a proper-subset hyperedge.

    Incidence overlap counting considers only retained edges and never enumerates
    all 2**|e| subsets. It can still be quadratic on dense adversarial inputs;
    this is a structural audit helper, not a claim of linear-time preprocessing.
    Reducing Boolean feasibility constraints does not preserve the continuous
    penalty polynomial, its gradients or the message-passing representation.
    """
    retained = []
    incidence = [[] for _ in range(problem.n)]
    removed = []
    incidence_visits = 0
    for original_index, edge in sorted(enumerate(problem.edges), key=lambda item: (len(item[1]), item[1])):
        overlaps = {}
        witness = None
        for vertex in edge:
            for retained_index in incidence[vertex]:
                incidence_visits += 1
                if len(retained[retained_index]) >= len(edge):
                    continue
                overlaps[retained_index] = overlaps.get(retained_index, 0) + 1
                if overlaps[retained_index] == len(retained[retained_index]):
                    witness = retained_index
                    break
            if witness is not None:
                break
        if witness is not None:
            removed.append({"original_index": original_index, "edge": list(edge),
                            "witness_edge": list(retained[witness])})
            continue
        retained_index = len(retained)
        retained.append(edge)
        for vertex in edge:
            incidence[vertex].append(retained_index)
    pruned = HypergraphProblem(problem.n, tuple(retained))
    retained_map = {edge: i for i, edge in enumerate(pruned.edges)}
    original_map = {edge: i for i, edge in enumerate(problem.edges)}
    kept_indices = [original_map[edge] for edge in pruned.edges]
    implied_by = [None] * len(problem.edges)
    original_to_pruned = [None] * len(problem.edges)
    for i, original_index in enumerate(kept_indices):
        original_to_pruned[original_index] = i
        implied_by[original_index] = i
    for item in removed:
        witness_index = retained_map[tuple(item["witness_edge"])]
        item["witness_kept_index"] = witness_index
        implied_by[item["original_index"]] = witness_index
    removed.sort(key=lambda item: item["original_index"])
    metadata = {
        "definition": "weak", "n": problem.n, "edge_index_base": 0,
        "original_edge_count": len(problem.edges), "retained_edge_count": len(pruned.edges),
        "removed_edge_count": len(removed), "kept_original_indices": kept_indices,
        "removed_original_indices": [item["original_index"] for item in removed],
        "original_to_pruned": original_to_pruned, "implied_by_pruned_index": implied_by,
        "removed_edges": removed, "incidence_visits": incidence_visits,
        "criterion": "remove an edge only when a retained edge is its strict subset",
        "boolean_weak_feasible_set_equivalent": True,
        "continuous_energy_equivalent": False,
        "scope": "Boolean weak constraints only; continuous loss and message structure can change",
    }
    return pruned, metadata
