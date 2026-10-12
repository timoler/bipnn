"""Small BIPNN-style MIS adaptation using public DHG HGNN+ layers.

Uses the public BIPNN polynomial-incidence loss idea. This is an adaptation,
not a faithful reproduction of all paper settings or a new architecture.
"""
import random
import time

import numpy as np
import torch
from torch import nn
from dhg import Hypergraph
from dhg.nn import HGNNPConv

from .problem import repair_and_extend


def polynomial_tensors(problem, definition, penalty, device):
    terms = problem.terms(definition, penalty)
    edges = list(terms)
    h = torch.zeros((problem.n, len(edges)), device=device)
    for j, edge in enumerate(edges):
        h[list(edge), j] = 1
    coefficients = torch.tensor(list(terms.values()), device=device)
    return edges, h, coefficients


def polynomial_loss(probabilities, incidence, coefficients):
    p = probabilities.reshape(-1, 1)
    factors = p * incidence + (1 - incidence)
    return (factors.prod(dim=0) * coefficients).sum()


class MISNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.conv1 = HGNNPConv(width, width, drop_rate=0.0, use_bn=False)
        self.conv2 = HGNNPConv(width, width, drop_rate=0.0, use_bn=False)
        self.head = nn.Linear(width, 1)

    def forward(self, features, graph):
        h = self.conv1(features, graph)
        h = self.conv2(h, graph)
        return torch.sigmoid(self.head(h)).flatten()


def solve_neural(problem, definition, *, seed=0, epochs=200, width=32,
                 lr=0.01, penalty=2.0, annealing=0.1, device="cpu"):
    if epochs < 1 or width < 1 or lr <= 0 or annealing < 0:
        raise ValueError("Invalid training settings")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    edges, incidence, coefficients = polynomial_tensors(problem, definition, penalty, device)
    # Singleton objective terms retain isolated vertices in the message-passing graph.
    graph = Hypergraph(problem.n, edges, device=device)
    features = torch.randn(problem.n, width, device=device)
    model = MISNet(width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        probabilities = model(features, graph)
        objective = polynomial_loss(probabilities, incidence, coefficients)
        binary_penalty = (1 - (2 * probabilities - 1).square()).sum()
        loss = objective + annealing * (epoch + 1) / epochs * binary_penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("Training produced a non-finite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Training produced a non-finite gradient")
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            trace.append({"epoch": epoch + 1, "loss_before_update": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        probabilities = model(features, graph)
        scores = probabilities.cpu().tolist()
        final_energy = float(polynomial_loss(probabilities, incidence, coefficients).cpu())
    raw = [i for i, p in enumerate(scores) if p >= 0.5]
    selected = repair_and_extend(problem, scores, definition)
    if not problem.feasible(selected, definition):
        raise AssertionError("Postprocessing returned an infeasible set")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "method": "hgnnp_pubo_repair", "seed": seed, "definition": definition,
        "selected": selected, "size": len(selected), "feasible": True,
        "raw_selected": raw, "raw_size": len(raw),
        "raw_feasible": problem.feasible(raw, definition),
        "scores": scores, "continuous_energy": final_energy,
        "total_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "settings": {"epochs": epochs, "width": width, "lr": lr,
                     "penalty": penalty, "annealing": annealing, "device": str(device)},
        "trace": trace,
    }
