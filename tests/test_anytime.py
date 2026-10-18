import unittest
import torch
from mis.anytime import empty_candidate, prefix_snapshot, random_fast_anytime, solve_fast_anytime, targets_checked
from mis.problem import HypergraphProblem


class AnytimeTests(unittest.TestCase):
    def test_prefix_never_uses_future_and_retains_earliest_tie(self):
        candidates = [empty_candidate(), {"size": 3, "available_seconds": 2.},
                      {"size": 4, "available_seconds": 5.001},
                      {"size": 4, "available_seconds": 7.}]
        a = prefix_snapshot(candidates, 5., 5.01)
        self.assertEqual(a["best_candidate_index"], 1)
        self.assertTrue(a["deadline_missed"])
        self.assertEqual(prefix_snapshot(candidates, 10., 10.01)["best_candidate_index"], 2)
        self.assertEqual(prefix_snapshot(candidates, 1., 1.01)["size"], 0)

    def test_targets_reject_ambiguity(self):
        for targets in [[], [0], [2, 1], [1, 1], [float("nan")]]:
            with self.assertRaises(ValueError):
                targets_checked(targets)

    def test_random_snapshots_are_strict_prefix_bests(self):
        problem = HypergraphProblem(12, ((0, 1, 2), (2, 3, 4), (5, 6)))
        result = random_fast_anytime(problem, "weak", 0, targets=[.005, .01, .02])
        self.assertEqual(len(result["records"]), 3)
        for record in result["records"]:
            prefix = [size for t, size in result["trial_log"] if t <= record["budget_seconds"]]
            self.assertEqual(record["size"], max([0] + prefix))
            self.assertEqual(record["trials"], len(prefix))
            self.assertLessEqual(record["best_available_seconds"], record["budget_seconds"])
            self.assertTrue(problem.feasible(record["selected"], "weak"))

    def test_cpu_training_has_valid_monotone_cutoffs(self):
        torch.set_num_threads(1)
        problem = HypergraphProblem(6, ((0, 1, 2), (2, 3, 4)))
        result = solve_fast_anytime(problem, "weak", "w256_cra", 0, targets=[.05, .1, .2], interval=.03, device="cpu")
        self.assertEqual(len(result["records"]), 3)
        sizes = [r["size"] for r in result["records"]]
        self.assertEqual(sizes, sorted(sizes))
        for record in result["records"]:
            rebuilt = prefix_snapshot(result["candidates"], record["budget_seconds"], record["total_seconds"])
            self.assertEqual(record["best_candidate_index"], rebuilt["best_candidate_index"])
            self.assertTrue(problem.feasible(record["selected"], "weak"))
        self.assertEqual(result["settings"]["horizon_seconds"], .2)


if __name__ == "__main__":
    unittest.main()
