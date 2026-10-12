import unittest
import torch
from mis.problem import HypergraphProblem
from mis.neural import polynomial_tensors, polynomial_loss, solve_neural


class NeuralTests(unittest.TestCase):
    def test_values_and_gradients_match_direct_polynomial(self):
        problem = HypergraphProblem(5, ((0, 1, 2), (2, 3), (0,)))
        for mode in ["weak", "strong"]:
            edges, h, c = polynomial_tensors(problem, mode, 2.0, "cpu")
            x = torch.tensor([0.2, 0.7, 0.4, 0.8, 0.6], dtype=torch.float64, requires_grad=True)
            actual = polynomial_loss(x, h.double(), c.double())
            direct = sum(coef * x[list(edge)].prod() for edge, coef in problem.terms(mode).items())
            self.assertTrue(torch.allclose(actual, direct))
            grad1 = torch.autograd.grad(actual, x, retain_graph=True)[0]
            grad2 = torch.autograd.grad(direct, x)[0]
            self.assertTrue(torch.allclose(grad1, grad2))

    def test_short_cpu_training(self):
        p = HypergraphProblem(5, ((0, 1, 2), (2, 3)))
        for mode in ["weak", "strong"]:
            r = solve_neural(p, mode, epochs=3, width=8)
            self.assertTrue(p.feasible(r["selected"], mode))
            self.assertIn(4, r["selected"])
            self.assertEqual(len(r["scores"]), p.n)


if __name__ == "__main__":
    unittest.main()
