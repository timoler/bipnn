import itertools
import random
import unittest

from mis.controlled import decode_stages
from mis.fast_decode import fast_decode_stages, fast_repair_and_extend
from mis.problem import HypergraphProblem, repair_and_extend


class FastDecodeTests(unittest.TestCase):
    def assert_same(self, problem, scores, mode):
        expected = decode_stages(problem, scores, mode)
        actual = fast_decode_stages(problem, scores, mode)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["selected"], repair_and_extend(problem, scores, mode))
        self.assertEqual(actual["selected"], fast_repair_and_extend(problem, scores, mode))
        self.assertEqual(actual["raw_feasible"], problem.feasible(actual["raw_selected"], mode))
        self.assertTrue(problem.feasible(actual["repaired_selected"], mode))
        self.assertTrue(problem.feasible(actual["selected"], mode))
        for v in set(range(problem.n)) - set(actual["selected"]):
            self.assertFalse(problem.feasible(actual["selected"] + [v], mode))

    def test_random_mixed_orders_with_ties_and_boundary_scores(self):
        rng = random.Random(2026101801)
        for _ in range(80):
            n = rng.randrange(1, 13)
            edges = tuple(tuple(rng.sample(range(n), rng.randrange(1, n + 1)))
                          for _ in range(rng.randrange(0, 20)))
            problem = HypergraphProblem(n, edges)
            score_vectors = [
                [rng.random() for _ in range(n)],
                [rng.choice([0., 0.499999, 0.5, 0.500001, 1.]) for _ in range(n)],
                [0.] * n, [0.5] * n, [1.] * n,
            ]
            for mode in ["weak", "strong"]:
                for scores in score_vectors:
                    self.assert_same(problem, scores, mode)

    def test_all_small_score_assignments_and_per_stage_equivalence(self):
        problem = HypergraphProblem(5, ((0,), (0, 1, 2), (1, 2), (2, 3, 4)))
        for scores in itertools.product([0., 0.5, 1.], repeat=problem.n):
            for mode in ["weak", "strong"]:
                self.assert_same(problem, scores, mode)

    def test_global_repair_priority_and_stale_heap_candidates(self):
        problem = HypergraphProblem(7, ((0, 1), (0, 2, 3), (2, 3), (4, 5)))
        scores = [0.9, 0.8, 0.6, 0.7, 0.8, 0.5, 0.5]
        for mode in ["weak", "strong"]:
            self.assert_same(problem, scores, mode)
            self.assertIn(6, fast_decode_stages(problem, scores, mode)["selected"])
        # Removing 0 satisfies both edges; stale candidates 1 and 2 must survive.
        stale = HypergraphProblem(3, ((0, 1), (0, 2)))
        self.assertEqual(fast_decode_stages(stale, [0.6, 0.8, 0.9], "weak")["repaired_selected"], [1, 2])

    def test_strong_edge_can_remain_violated_after_several_removals(self):
        problem = HypergraphProblem(6, ((0, 1, 2, 3, 4),))
        for mode in ["weak", "strong"]:
            self.assert_same(problem, [0.8] * 6, mode)
        self.assertEqual(fast_decode_stages(problem, [0.8] * 6, "strong")["repaired_selected"], [4, 5])

    def test_singletons_isolates_and_no_edges(self):
        for problem in [HypergraphProblem(4, ((0,), (1, 2))),
                        HypergraphProblem(3, ((0,), (1,), (2,))),
                        HypergraphProblem(4, ())]:
            for mode in ["weak", "strong"]:
                for scores in [[0.] * problem.n, [0.5] * problem.n, [1.] * problem.n]:
                    self.assert_same(problem, scores, mode)
        problem = HypergraphProblem(2, ((0,),))
        self.assertEqual(fast_decode_stages(problem, [1., 0.], "weak")["selected"], [1])
        self.assertEqual(fast_decode_stages(problem, [1., 0.], "strong")["selected"], [0, 1])

    def test_finite_scores_outside_probability_interval_still_match(self):
        problem = HypergraphProblem(4, ((0, 1, 2), (1, 2, 3)))
        for mode in ["weak", "strong"]:
            self.assert_same(problem, [-20., 0.5, 1., 10.], mode)

    def test_invalid_definition_length_and_nonfinite_scores(self):
        problem = HypergraphProblem(3, ((0, 1, 2),))
        for decoder in [decode_stages, fast_decode_stages]:
            for mode in [None, "", "unknown"]:
                with self.assertRaises(ValueError):
                    decoder(problem, [0.5] * 3, mode)
            for scores in [[], [0.5], [0.] * 4, [0., float("nan"), 1.],
                           [0., float("inf"), 1.], [0., -float("inf"), 1.]]:
                for mode in ["weak", "strong"]:
                    with self.assertRaises(ValueError):
                        decoder(problem, scores, mode)


if __name__ == "__main__":
    unittest.main()
