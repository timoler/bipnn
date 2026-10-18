"""New fast-decoder anytime protocol; prior controlled sources remain frozen."""
import bisect
import math
import random
import time

import numpy as np
import torch
from dhg import Hypergraph

from .controlled import CONFIGS
from .fast_decode import fast_decode_stages
from .indexed_loss import indexed_tensors, indexed_polynomial_loss
from src.models import Layer, LayerType, Net
from src.train import train


def targets_checked(targets):
    values = [float(t) for t in targets]
    if not values or values != sorted(set(values)) or any(not math.isfinite(t) or t <= 0 for t in values):
        raise ValueError("Targets must be finite, positive and strictly increasing")
    return values


def empty_candidate():
    return {"selected": [], "size": 0, "feasible": True, "available_seconds": 0.,
            "role": "empty_initial_incumbent", "epoch": 0, "trial": 0}


def prefix_snapshot(candidates, deadline, emitted):
    """Only completed candidates at/before cutoff; earliest wins ties."""
    eligible = [i for i, candidate in enumerate(candidates) if candidate["available_seconds"] <= deadline]
    if not eligible:
        raise ValueError("Missing initial feasible incumbent")
    best = max(eligible, key=lambda i: candidates[i]["size"])
    return {**candidates[best], "budget_seconds": deadline, "cutoff_seconds": deadline,
            "total_seconds": emitted, "snapshot_emitted_seconds": emitted,
            "overshoot_seconds": max(0., emitted - deadline), "deadline_missed": emitted > deadline,
            "best_candidate_index": best, "latest_candidate_index": eligible[-1],
            "candidate_count": len(eligible), "best_available_seconds": candidates[best]["available_seconds"]}


def solve_fast_anytime(problem, mode, config, seed, *, targets=(5, 15, 30), interval=1., device="cuda"):
    targets = targets_checked(targets)
    if config not in CONFIGS or not math.isfinite(interval) or interval <= 0:
        raise ValueError("Invalid configuration or interval")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    edges, indices, coefficients = indexed_tensors(problem, mode, 2., device)
    graph = Hypergraph(problem.n, edges, device=device)
    width = CONFIGS[config]["width"]
    net = Net([Layer(LayerType.HGNNPCONV, width, width),
               Layer(LayerType.HGNNPCONV, width, width // 2, last_conv=True),
               Layer(LayerType.LINEAR, width // 2, 1, dropout=0.)]).to(device)
    features = torch.rand(problem.n, width, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-8)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    built = time.perf_counter()
    horizon = targets[-1]
    due = sorted({i * interval for i in range(1, math.ceil(horizon / interval))} |
                 {t - min(.1, t / 10) for t in targets})
    candidates, snapshots = [empty_candidate()], []
    epochs, train_seconds, decode_seconds, due_index = 0, 0., 0., 0
    initial_index = None

    def collect(outs, role, loss, coefficient):
        nonlocal decode_seconds, initial_index
        begin = time.perf_counter()
        scores = outs.detach().flatten().cpu().tolist()
        decoded = fast_decode_stages(problem, scores, mode)
        available = time.perf_counter() - start
        candidates.append({**decoded, "scores": scores, "role": role, "epoch": epochs,
                           "available_seconds": available, "loss_before_update": loss,
                           "annealing_coefficient": coefficient})
        if role == "untrained_first_forward":
            initial_index = len(candidates) - 1
        decode_seconds += time.perf_counter() - begin

    def emit_ready():
        while len(snapshots) < len(targets) and time.perf_counter() - start >= targets[len(snapshots)]:
            target = targets[len(snapshots)]
            record = prefix_snapshot(candidates, target, time.perf_counter() - start)
            record.update({"method": "bipnn_fast_anytime", "seed": seed, "config": config,
                           "epochs_completed_at_snapshot": epochs,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
            snapshots.append(record)

    while time.perf_counter() - start < horizon or epochs == 0:
        elapsed = time.perf_counter() - start
        coefficient = (-1 + 1.6 * min(1., elapsed / horizon)) if CONFIGS[config]["annealing"] == "cra" else 0.

        def objective(outs, **kwargs):
            energy = indexed_polynomial_loss(outs, indices, coefficients)
            return energy + coefficient * (1 - (2 * outs - 1).square()).sum()

        training_start = time.perf_counter()
        loss, outs = train(net, features, graph, optimizer, objective)
        train_seconds += time.perf_counter() - training_start
        epochs += 1
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite anytime loss")
        now = time.perf_counter() - start
        if epochs == 1:
            collect(outs, "untrained_first_forward", loss, coefficient)
        elif due_index < len(due) and now >= due[due_index]:
            collect(outs, "periodic_training_forward", loss, coefficient)
        while due_index < len(due) and due[due_index] <= time.perf_counter() - start:
            due_index += 1
        emit_ready()
    emit_ready()
    # Later diagnostic output never enters an earlier cutoff's selected prefix.
    collect(outs, "terminal_diagnostic", loss, coefficient)
    emit_ready()
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]).all()):
        raise FloatingPointError("Nonfinite final gradient")
    total = time.perf_counter() - start
    return {"method": "bipnn_fast_anytime", "seed": seed, "config": config, "records": snapshots,
            "candidates": candidates, "initial_index": initial_index,
            "last_index": len(candidates) - 1, "epochs_completed": epochs,
            "run_total_seconds": total, "build_seconds": built - start,
            "training_seconds": train_seconds, "online_decode_seconds": decode_seconds,
            "settings": {**CONFIGS[config], "targets_seconds": targets, "horizon_seconds": horizon,
                         "candidate_interval_seconds": interval, "pre_cutoff_margin_seconds": .1,
                         "penalty": 2., "lr": 1e-4, "weight_decay": 5e-8,
                         "effective_dropout": 0., "output": "pre-update training forward",
                         "selection": "first maximum feasible candidate completed by cutoff",
                         "schedule": "-1+1.6*min(1,elapsed/horizon) if CRA, else 0"}}


def random_fast_anytime(problem, mode, seed, *, targets=(5, 15, 30)):
    targets = targets_checked(targets)
    rng = random.Random(seed)
    candidates, trial_log, snapshots = [empty_candidate()], [], []
    start = time.perf_counter()
    best_size, first = 0, None

    def emit_ready():
        while len(snapshots) < len(targets) and time.perf_counter() - start >= targets[len(snapshots)]:
            target = targets[len(snapshots)]
            record = prefix_snapshot(candidates, target, time.perf_counter() - start)
            record.update({"method": "random_restart_same_decoder", "seed": seed,
                           "trials": bisect.bisect_right([r[0] for r in trial_log], target)})
            snapshots.append(record)

    while time.perf_counter() - start < targets[-1] or not trial_log:
        scores = [rng.random() for _ in range(problem.n)]
        decoded = fast_decode_stages(problem, scores, mode)
        available = time.perf_counter() - start
        trial = len(trial_log) + 1
        trial_log.append([available, decoded["size"]])
        candidate = {**decoded, "scores": scores, "available_seconds": available,
                     "trial": trial, "role": "random_trial"}
        if first is None:
            first = {**candidate, "method": "random_score_same_decoder", "seed": seed,
                     "total_seconds": available, "budget_seconds": None, "trials": 1}
        if decoded["size"] > best_size:
            candidates.append(candidate)
            best_size = decoded["size"]
        emit_ready()
    emit_ready()
    return {"method": "random_restart_same_decoder", "seed": seed, "records": snapshots,
            "single": first, "candidates": candidates, "trial_log": trial_log,
            "run_total_seconds": time.perf_counter() - start}
