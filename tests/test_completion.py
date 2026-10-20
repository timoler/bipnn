import itertools
import json
import math
import unittest
from unittest.mock import patch

import torch

from mis.completion import (solve_fixed_epochs, solve_timed_control, prune_weak_constraints,
                            _prefix_snapshot, _empty_candidate)
from mis.fast_decode import fast_decode_stages
from mis.problem import HypergraphProblem


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_candidate(self, problem, mode, candidate):
        self.assertTrue(problem.feasible(candidate["selected"], mode))
        for key, value in fast_decode_stages(problem, candidate["scores"], mode).items():
            self.assertEqual(candidate[key], value)
        self.assertEqual(len(candidate["scores"]), problem.n)
        self.assertTrue(math.isfinite(candidate["loss_before_update"]))
        self.assertGreaterEqual(candidate["available_seconds"], 0.)

    def test_pruning_preserves_every_binary_weak_assignment_and_explicit_n(self):
        problems = [
            HypergraphProblem(5, ()),
            HypergraphProblem(1, ((0,),)),
            HypergraphProblem(6, ((0,), (0, 1), (0, 1, 2), (1, 2), (1, 2, 3), (3, 4))),
            HypergraphProblem(5, ((0, 1), (1, 2), (0, 1, 2), (2, 3, 4))),
        ]
        for problem in problems:
            pruned, metadata = prune_weak_constraints(problem)
            self.assertEqual(pruned.n, problem.n)
            for bits in itertools.product([0, 1], repeat=problem.n):
                selected = [i for i, bit in enumerate(bits) if bit]
                self.assertEqual(problem.feasible(selected, "weak"), pruned.feasible(selected, "weak"))
            for item in metadata["removed_edges"]:
                witness = pruned.edges[item["witness_kept_index"]]
                self.assertLess(len(witness), len(item["edge"]))
                self.assertTrue(set(witness) < set(item["edge"]))
                self.assertEqual(list(witness), item["witness_edge"])
            self.assertFalse(metadata["continuous_energy_equivalent"])
            self.assertEqual(metadata["retained_edge_count"] + metadata["removed_edge_count"], len(problem.edges))
            json.dumps(metadata, allow_nan=False)

    def test_pruning_singletons_and_large_edges_without_subset_enumeration(self):
        problem = HypergraphProblem(70, ((0,), tuple(range(60)), tuple(range(1, 65))))
        pruned, metadata = prune_weak_constraints(problem)
        self.assertEqual(pruned.n, 70)
        self.assertEqual(pruned.edges, ((0,), tuple(range(1, 65))))
        self.assertEqual(metadata["removed_edge_count"], 1)
        self.assertEqual(metadata["original_to_pruned"], [0, None, 1])
        self.assertEqual(metadata["implied_by_pruned_index"], [0, 0, 1])
        self.assertIn(69, [v for v in range(pruned.n) if all(v not in e for e in pruned.edges)])

    def test_fixed_checkpoints_schedule_stages_and_first_maximum(self):
        problem = HypergraphProblem(6, ((0,), (1, 2, 3), (2, 4)))
        for mode in ["weak", "strong"]:
            result = solve_fixed_epochs(problem, mode, 3, epochs=5, width=8,
                                        checkpoint_every=2, annealing="cra", device="cpu")
            self.assertEqual([c["epoch"] for c in result["candidates"]], [1, 2, 4, 5])
            self.assertEqual(result["candidates"][0]["gamma"], -1.)
            self.assertAlmostEqual(result["candidates"][-1]["gamma"], .6)
            self.assertEqual(result["initial"]["role"], "untrained_first_forward")
            self.assertEqual([c["role"] for c in result["candidates"]],
                             ["untrained_first_forward", "epoch_checkpoint", "epoch_checkpoint", "final_epoch"])
            self.assertEqual(result["epochs_completed"], 5)
            for candidate in result["candidates"]:
                self.assert_candidate(problem, mode, candidate)
                self.assertEqual(candidate["coefficient_progress"], (candidate["epoch"] - 1) / 4)
            expected = max(range(len(result["candidates"])), key=lambda i: result["candidates"][i]["size"])
            self.assertEqual(result["best_index"], expected)
            self.assertEqual(result["latest"]["epoch"], 5)
            self.assertGreaterEqual(result["total_seconds"], result["build_seconds"])
            self.assertEqual(len(result["initial_state_sha256"]), 64)
            self.assertEqual(len(result["feature_sha256"]), 64)
            json.dumps(result, allow_nan=False)

    def test_fixed_one_step_no_duplicate_initial_and_none_schedule(self):
        problem = HypergraphProblem(4, ())
        for annealing, gamma in [("cra", -1.), ("none", 0.)]:
            result = solve_fixed_epochs(problem, "weak", 1, epochs=1, width=8,
                                        checkpoint_every=1, annealing=annealing, device="cpu")
            self.assertEqual(len(result["candidates"]), 1)
            self.assertEqual(result["best_index"], 0)
            self.assertEqual(result["latest_index"], 0)
            self.assertEqual(result["initial"]["gamma"], gamma)
            self.assertEqual(result["best"]["selected"], [0, 1, 2, 3])

    def test_fixed_equal_size_checkpoints_keep_first_candidate(self):
        problem = HypergraphProblem(4, ())
        result = solve_fixed_epochs(problem, "weak", 1, epochs=3, width=8,
                                    checkpoint_every=1, annealing="none", device="cpu")
        self.assertEqual([c["epoch"] for c in result["candidates"]], [1, 2, 3])
        self.assertEqual([c["size"] for c in result["candidates"]], [4, 4, 4])
        self.assertEqual(result["best_index"], 0)
        self.assertEqual(result["latest_index"], 2)

    def test_initial_hashes_match_across_penalty_and_annealing_without_cross_run_reuse(self):
        problem = HypergraphProblem(5, ((0, 1, 2), (2, 3)))
        results = [solve_fixed_epochs(problem, "weak", 8, epochs=2, width=8,
                                     penalty=penalty, annealing=annealing, device="cpu")
                   for penalty, annealing in [(1.1, "cra"), (2., "none"), (4., "cra"), (8., "none")]]
        self.assertEqual(len({r["initial_state_sha256"] for r in results}), 1)
        self.assertEqual(len({r["feature_sha256"] for r in results}), 1)
        self.assertEqual(len({tuple(r["initial"]["scores"]) for r in results}), 1)
        changed_seed = solve_fixed_epochs(problem, "weak", 9, epochs=1, width=8, device="cpu")
        self.assertNotEqual(changed_seed["initial_state_sha256"], results[0]["initial_state_sha256"])
        self.assertNotEqual(changed_seed["feature_sha256"], results[0]["feature_sha256"])

    def test_objective_and_message_can_differ_but_decode_uses_original_problem(self):
        problem = HypergraphProblem(6, ((0,), (1, 2), (1, 2, 3), (2, 4, 5)))
        objective, _ = prune_weak_constraints(problem)
        message = HypergraphProblem(6, ())
        a = solve_fixed_epochs(problem, "weak", 2, epochs=2, width=8,
                               objective_problem=objective, message_problem=message, device="cpu")
        b = solve_fixed_epochs(problem, "weak", 2, epochs=2, width=8,
                               objective_problem=problem, message_problem=message, device="cpu")
        self.assertNotEqual(a["settings"]["objective_monomial_count"], a["settings"]["message_monomial_count"])
        self.assertEqual(a["settings"]["message_monomial_count"], 6)
        self.assertEqual(a["settings"]["decode_constraint_count"], len(problem.edges))
        self.assertEqual(a["initial"]["scores"], b["initial"]["scores"])
        self.assertNotEqual(a["initial"]["loss_before_update"], b["initial"]["loss_before_update"])
        for candidate in a["candidates"]:
            self.assert_candidate(problem, "weak", candidate)
            self.assertNotIn(0, candidate["selected"])
        with self.assertRaises(ValueError):
            solve_fixed_epochs(problem, "weak", 2, epochs=1, width=8,
                               message_problem=HypergraphProblem(5, ()), device="cpu")

    def test_prefix_snapshot_excludes_future_and_keeps_first_tie(self):
        candidates = [_empty_candidate(),
                      {"selected": [0], "size": 1, "available_seconds": .1},
                      {"selected": [1], "size": 1, "available_seconds": .2},
                      {"selected": [0, 1], "size": 2, "available_seconds": .31}]
        result = _prefix_snapshot(candidates, .3, .4)
        self.assertEqual(result["selected"], [0])
        self.assertEqual(result["best_candidate_index"], 1)
        self.assertEqual(result["latest_candidate_index"], 2)
        self.assertEqual(result["candidate_count"], 3)
        self.assertAlmostEqual(result["overshoot_seconds"], .1)

    def test_short_cpu_timed_control_has_complete_prefix_and_late_terminal(self):
        problem = HypergraphProblem(5, ((0,), (1, 2, 3), (3, 4)))
        results = []
        for annealing in ["cra", "none"]:
            result = solve_timed_control(problem, "weak", 4, annealing=annealing,
                                         targets=(.01, .02), interval=.005, width=8, device="cpu")
            results.append(result)
            self.assertEqual(len(result["records"]), 2)
            self.assertEqual(result["settings"]["horizon_seconds"], .02)
            self.assertEqual(result["candidates"][0]["role"], "empty_initial_incumbent")
            self.assertEqual(result["initial"]["role"], "untrained_first_forward")
            self.assertEqual(result["terminal"]["role"], "terminal_diagnostic")
            for candidate in result["candidates"][1:]:
                self.assert_candidate(problem, "weak", candidate)
                elapsed = candidate["coefficient_elapsed_seconds"]
                expected_gamma = -1 + 1.6 * min(1., elapsed / .02) if annealing == "cra" else 0.
                self.assertAlmostEqual(candidate["gamma"], expected_gamma)
                if annealing == "none":
                    self.assertEqual(candidate["gamma"], 0.)
            for record in result["records"]:
                self.assertLessEqual(record["available_seconds"], record["cutoff_seconds"])
                expected = _prefix_snapshot(result["candidates"], record["cutoff_seconds"], record["total_seconds"])
                self.assertEqual(record["selected"], expected["selected"])
                self.assertEqual(record["best_candidate_index"], expected["best_candidate_index"])
            self.assertGreater(result["terminal"]["available_seconds"], .02)
            json.dumps(result, allow_nan=False)
        self.assertEqual(results[0]["initial_state_sha256"], results[1]["initial_state_sha256"])
        self.assertEqual(results[0]["feature_sha256"], results[1]["feature_sha256"])

    def test_invalid_settings_and_nonfinite_training_loss(self):
        problem = HypergraphProblem(4, ((0, 1, 2),))
        for settings in [{"epochs": 0}, {"epochs": True}, {"checkpoint_every": 0},
                         {"width": 3}, {"annealing": "unknown"}, {"penalty": 1.},
                         {"penalty": math.inf}]:
            with self.assertRaises(ValueError):
                solve_fixed_epochs(problem, "weak", 0, device="cpu", **settings)
        for targets in [(), (0,), (2, 1), (1, 1), (math.nan,)]:
            with self.assertRaises(ValueError):
                solve_timed_control(problem, "weak", 0, annealing="cra", targets=targets, device="cpu")
        with self.assertRaises(ValueError):
            solve_fixed_epochs(HypergraphProblem(1, ()), "weak", 0, epochs=1, device="cpu")
        with patch("mis.completion.train", return_value=(math.nan, torch.zeros(4, 1))):
            with self.assertRaises(FloatingPointError):
                solve_fixed_epochs(problem, "weak", 0, epochs=1, width=8, device="cpu")


if __name__ == "__main__":
    unittest.main()
