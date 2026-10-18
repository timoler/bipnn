import unittest
from mis.benchmark import generate_instance, random_decoder
from mis.problem import HypergraphProblem
from mis.neural import solve_neural


class BenchmarkTests(unittest.TestCase):
    def test_generator_is_fixed_and_retains_declared_nodes(self):
        a = generate_instance(20, 3, 2, 42)
        self.assertEqual(a, generate_instance(20, 3, 2, 42))
        self.assertNotEqual(a["edges"], generate_instance(20, 3, 2, 43)["edges"])
        self.assertEqual(len(a["edges"]), 40)
        self.assertEqual(a["n"], 20)
        self.assertEqual(len(set(map(tuple, a["edges"]))), 40)

    def test_random_controls_save_their_actual_best_solution(self):
        p = HypergraphProblem(8, ((0,1,2), (2,3,4), (4,5,6)))
        for mode in ["weak", "strong"]:
            r = random_decoder(p, mode, 0, budget=0.001)
            self.assertTrue(p.feasible(r["selected"], mode))
            self.assertIn(7, r["selected"])
            self.assertEqual(len(r["selected"]), r["size"])

    def test_time_budget_stops_and_keeps_valid_output(self):
        p = HypergraphProblem(5, ((0,1,2), (2,3)))
        r = solve_neural(p, "weak", epochs=10000, width=8, time_limit=0.001)
        self.assertGreaterEqual(r["epochs_completed"], 1)
        self.assertLess(r["epochs_completed"], 10000)
        self.assertTrue(p.feasible(r["selected"], "weak"))


if __name__ == "__main__":
    unittest.main()
