"""Controlled public-architecture adaptation; existing experiment files stay frozen."""
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from dhg import Hypergraph

from .benchmark import write_json
from .problem import check_definition
from src.models import Layer, LayerType, Net
from src.train import train


CONFIGS = {
    "w256_cra": {"width": 256, "annealing": "cra"},
    "w256_none": {"width": 256, "annealing": "none"},
    "w1024_cra": {"width": 1024, "annealing": "cra"},
    "w1024_none": {"width": 1024, "annealing": "none"},
}


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def bind_manifest(output, protocol):
    """Refuse silent resumes after any relevant source/configuration change."""
    root = Path(__file__).resolve().parents[1]
    paths = ["mis/controlled.py", "mis/indexed_loss.py", "mis/problem.py",
             "mis/benchmark.py", "mis/reference.py", "src/models.py", "src/train.py",
             "scripts/run_controlled.py"]
    bound = {"protocol": protocol,
             "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}}
    key = fingerprint(bound)
    path = Path(output) / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8"))
        if old["binding"] != key:
            raise ValueError("Resume refused: protocol or source hashes differ")
        return old
    manifest = {**bound, "binding": key,
                "git_commit_at_start": subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                                "cpu_threads": torch.get_num_threads()}}
    write_json(path, manifest)
    return manifest


def decode_stages(problem, scores, mode):
    """Expose exactly the threshold/repair/extension stages of the old decoder."""
    check_definition(mode)
    if len(scores) != problem.n or any(not math.isfinite(p) for p in scores):
        raise ValueError("One finite score is required per vertex")
    raw = [i for i, p in enumerate(scores) if p >= 0.5]
    selected = set(raw)
    while True:
        bad = problem.violations(selected, mode)
        if not bad:
            break
        candidates = {v for edge in bad for v in edge if v in selected}
        selected.remove(min(candidates, key=lambda v: (scores[v], v)))
    repaired = sorted(selected)
    for v in sorted(range(problem.n), key=lambda v: (-scores[v], v)):
        if v not in selected and problem.feasible(selected | {v}, mode):
            selected.add(v)
    return {"raw_selected": raw, "raw_size": len(raw), "raw_feasible": problem.feasible(raw, mode),
            "repaired_selected": repaired, "repaired_size": len(repaired),
            "selected": sorted(selected), "size": len(selected), "feasible": True,
            "removed_by_repair": len(raw) - len(repaired),
            "added_by_extension": len(selected) - len(repaired)}


def solve_controlled(problem, mode, config, seed, *, budget=5.0, device="cuda", epochs=None):
    """CRA uses wall-clock progress for budgets, epoch/1000 for fixed epochs.

    Timed runs include model/objective construction and final decoding. The
    stopping decision is made between training steps, so some overshoot is
    unavoidable and is saved. Initial/checkpoint decoding is diagnostic and
    separately timed after the primary solution is finalized.
    """
    from .indexed_loss import indexed_tensors, indexed_polynomial_loss
    if config not in CONFIGS or not math.isfinite(budget) or budget <= 0:
        raise ValueError("Invalid configuration or budget")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
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
    edges, indices, coefficients = indexed_tensors(problem, mode, 2.0, device)
    graph = Hypergraph(problem.n, edges, device=device)
    width = CONFIGS[config]["width"]
    net = Net([Layer(LayerType.HGNNPCONV, width, width),
               Layer(LayerType.HGNNPCONV, width, width // 2, last_conv=True),
               Layer(LayerType.LINEAR, width // 2, 1, dropout=0.0)]).to(device)
    features = torch.rand(problem.n, width, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-8)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    built = time.perf_counter()
    trace, initial_scores = [], None
    next_fraction = 0.25
    completed = 0
    while completed < (epochs if epochs is not None else 1000000):
        elapsed = time.perf_counter() - start
        if completed and epochs is None and elapsed >= budget:
            break
        progress = ((completed + 1) / 1600 if epochs is not None else min(1., elapsed / budget))
        coefficient = -1 + 1.6 * progress if CONFIGS[config]["annealing"] == "cra" else 0.0

        def objective(outs, **kwargs):
            pubo = indexed_polynomial_loss(outs, indices, coefficients)
            return pubo + coefficient * (1 - (2 * outs - 1).square()).sum()

        loss, outs = train(net, features, graph, optimizer, objective)
        completed += 1
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite training objective")
        if completed == 1:
            initial_scores = outs.detach().flatten().cpu().tolist()
        fraction = (completed / epochs if epochs is not None else
                    (time.perf_counter() - start) / budget)
        if fraction >= next_fraction:
            trace.append({"epoch": completed, "seconds": time.perf_counter() - start,
                          "loss_before_update": loss, "annealing_coefficient": coefficient,
                          "scores": outs.detach().flatten().cpu().tolist()})
            next_fraction += 0.25
    # Match the public runner's last pre-update training output, not eval().
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in net.parameters()
                             if p.grad is not None]).all()):
        raise FloatingPointError("Nonfinite final gradient")
    scores = outs.detach().flatten().cpu().tolist()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trained = time.perf_counter()
    result = decode_stages(problem, scores, mode)
    primary_end = time.perf_counter()
    result.update({"method": "public_architecture_indexed_budget", "config": config, "seed": seed,
                   "scores": scores, "total_seconds": primary_end - start,
                   "build_seconds": built - start, "training_seconds": trained - built,
                   "decode_seconds": primary_end - trained, "epochs_completed": completed,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "settings": {**CONFIGS[config], "lr": 1e-4, "weight_decay": 5e-8,
                                "penalty": 2.0, "effective_dropout": 0.0,
                                "budget_seconds": budget, "fixed_epochs": epochs,
                                "cra_schedule": "-1+1.6*(elapsed setup+train / budget)" if epochs is None else "-1+epoch/1000",
                                "output": "last training-mode forward before final update"}})
    initial = decode_stages(problem, initial_scores, mode)
    result["initial"] = {**initial, "scores": initial_scores}
    for checkpoint in trace:
        checkpoint.update(decode_stages(problem, checkpoint["scores"], mode))
    result["checkpoints"] = trace
    result["diagnostic_decode_seconds"] = time.perf_counter() - primary_end
    return result
