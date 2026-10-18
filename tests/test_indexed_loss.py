import unittest

import torch

from mis.indexed_loss import indexed_polynomial_loss, indexed_tensors
from mis.neural import polynomial_loss, polynomial_tensors
from mis.problem import HypergraphProblem


class IndexedLossTests(unittest.TestCase):
    def setUp(self):
        # Mixed orders, a singleton constraint, and isolated vertices 6 and 7.
        self.problem = HypergraphProblem(
            8, ((0,), (1, 2), (2, 3, 4), (1, 2, 3, 5)))

    def assert_equivalent(self, problem, definition, values):
        dense_edges, incidence, dense_coefs = polynomial_tensors(
            problem, definition, 2.0, "cpu")
        edges, indices, coefficients = indexed_tensors(
            problem, definition, 2.0, "cpu")
        self.assertEqual(edges, dense_edges)
        self.assertEqual(edges, list(problem.terms(definition, 2.0)))
        self.assertTrue(torch.equal(coefficients, dense_coefs))
        self.assertTrue(torch.all((indices >= 0) & (indices <= problem.n)))
        p = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        dense = polynomial_loss(p, incidence.double(), dense_coefs.double())
        indexed = indexed_polynomial_loss(p, indices, coefficients.double())
        self.assertEqual(indexed.ndim, 0)
        torch.testing.assert_close(indexed, dense, rtol=1e-12, atol=1e-12)
        dense_grad = torch.autograd.grad(dense, p)[0]
        indexed_grad = torch.autograd.grad(indexed, p)[0]
        torch.testing.assert_close(indexed_grad, dense_grad, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.isfinite(indexed_grad).all())

    def test_mixed_orders_values_and_gradients_at_boundaries(self):
        values = [
            [0.2, 0.7, 0.4, 0.8, 0.6, 0.3, 0.1, 0.9],
            [0.0] * 8,
            [1.0] * 8,
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ]
        for definition in ("weak", "strong"):
            for vector in values:
                with self.subTest(definition=definition, values=vector):
                    self.assert_equivalent(self.problem, definition, vector)

    def test_all_isolated_vertices(self):
        problem = HypergraphProblem(4, ())
        for definition in ("weak", "strong"):
            with self.subTest(definition=definition):
                self.assert_equivalent(problem, definition, [0.0, 0.2, 0.8, 1.0])

    def test_gradcheck_nonboundary(self):
        generator = torch.Generator().manual_seed(516)
        for definition in ("weak", "strong"):
            _, indices, coefficients = indexed_tensors(
                self.problem, definition, 2.0, "cpu")
            p = (0.1 + 0.8 * torch.rand(
                self.problem.n, dtype=torch.float64, generator=generator)).requires_grad_()
            with self.subTest(definition=definition):
                self.assertTrue(torch.autograd.gradcheck(
                    lambda x: indexed_polynomial_loss(x, indices, coefficients.double()),
                    (p,), eps=1e-6, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
