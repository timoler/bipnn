"""Index-based evaluation of the existing MIS polynomial.

Padding uses a constant-one sentinel, so ordinary ``prod`` handles zero-valued
probabilities and their derivatives. This changes storage and computation only;
it is not a new objective or a claim of identical floating-point training paths.
"""
import torch


def indexed_tensors(problem, definition, penalty, device):
    """Return terms in exactly ``problem.terms`` order and padded node indices.

    Index ``problem.n`` is reserved for a constant one appended at evaluation.
    Coefficients use the same default dtype as ``mis.neural.polynomial_tensors``.
    """
    terms = problem.terms(definition, penalty)
    edges = list(terms)
    max_order = max(map(len, edges))
    padded = [list(edge) + [problem.n] * (max_order - len(edge)) for edge in edges]
    indices = torch.tensor(padded, dtype=torch.long, device=device)
    coefficients = torch.tensor(list(terms.values()), device=device)
    return edges, indices, coefficients


def indexed_polynomial_loss(probabilities, indices, coefficients):
    """Evaluate the unchanged scalar polynomial with differentiable gathering."""
    p = probabilities.reshape(-1)
    padded = torch.cat((p, p.new_ones(1)))
    return (padded[indices].prod(dim=1) * coefficients).sum()
