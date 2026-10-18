import json
import math
import random
import unittest
from unittest.mock import patch

from mis.fast_baselines import (fast_degree_greedy, random_fast, solve_scip_fast,
                                _json_scip_number, _cutoff_snapshot)
from mis.fast_decode import fast_decode_stages
from mis.problem import HypergraphProblem, degree_greedy, exact_mis


class FastBaselineTests(unittest.TestCase):
    def assert_same_degree(self, problem, mode):
        selected = fast_degree_greedy(problem, mode)
        self.assertEqual(selected, degree_greedy(problem, mode))
        self.assertTrue(problem.feasible(selected, mode))
        for vertex in set(range(problem.n)) - set(selected):
            self.assertFalse(problem.feasible(selected + [vertex], mode))

    def test_degree_random_mixed_edges_exact_selected_equivalence(self):
        rng = random.Random(2026101802)
        for _ in range(120):
            n = rng.randrange(1, 17)
            edges = tuple(tuple(rng.sample(range(n), rng.randrange(1, n + 1)))
                          for _ in range(rng.randrange(0, 25)))
            problem = HypergraphProblem(n, edges)
            for mode in ["weak", "strong"]:
                self.assert_same_degree(problem, mode)

    def test_degree_singletons_isolates_empty_edges_and_ties(self):
        for problem in [HypergraphProblem(4, ()),
                        HypergraphProblem(5, ((0,), (1, 2))),
                        HypergraphProblem(4, ((0,), (1,), (2,), (3,))),
                        HypergraphProblem(6, ((0, 1), (1, 2), (2, 3), (3, 0))),
                        HypergraphProblem(5, ((0, 1, 2, 3),))]:
            for mode in ["weak", "strong"]:
                self.assert_same_degree(problem, mode)
        # Every non-isolate has the same original hyperedge degree; vertex ID breaks ties.
        tied = HypergraphProblem(5, ((0, 1, 2, 3),))
        self.assertEqual(fast_degree_greedy(tied, "weak"), [0, 1, 2, 4])
        self.assertEqual(fast_degree_greedy(tied, "strong"), [0, 4])

    def test_random_single_preserves_actual_score_and_stages(self):
        problem = HypergraphProblem(8, ((0,), (1, 2, 3), (2, 4), (4, 5, 6)))
        for mode in ["weak", "strong"]:
            result = random_fast(problem, mode, 9)
            rng = random.Random(9)
            scores = [rng.random() for _ in range(problem.n)]
            self.assertEqual(result["scores"], scores)
            self.assertEqual(result["trials"], 1)
            self.assertEqual(result["method"], "random_score_same_decoder")
            for key, expected in fast_decode_stages(problem, scores, mode).items():
                self.assertEqual(result[key], expected)
            json.dumps(result, allow_nan=False)

    def test_random_restart_retains_best_complete_trial_and_earliest_tie(self):
        problem = HypergraphProblem(8, ((0,), (1, 2, 3), (2, 4), (4, 5, 6)))
        for mode in ["weak", "strong"]:
            rng = random.Random(4)
            candidates = []
            for _ in range(3):
                scores = [rng.random() for _ in range(problem.n)]
                candidates.append({**fast_decode_stages(problem, scores, mode), "scores": scores})
            expected = max(candidates, key=lambda r: r["size"])
            with patch("mis.fast_baselines.time.perf_counter", side_effect=[0., .1, .2, .3, .31]):
                result = random_fast(problem, mode, 4, budget=.25)
            self.assertEqual(result["trials"], 3)
            self.assertEqual(result["total_seconds"], .31)
            self.assertEqual(result["method"], "random_restart_same_decoder")
            for key, value in expected.items():
                self.assertEqual(result[key], value)

    def test_scip_matches_enumeration_and_safe_json_metadata(self):
        for problem in [HypergraphProblem(7, ((0, 1, 2), (2, 3, 4), (4, 5))),
                        HypergraphProblem(4, ((0,), (1, 2))), HypergraphProblem(4, ())]:
            for mode in ["weak", "strong"]:
                result = solve_scip_fast(problem, mode, time_limit=.5)
                optimum = len(exact_mis(problem, mode))
                self.assertTrue(result["optimality_proven"])
                self.assertEqual(result["size"], optimum)
                self.assertGreaterEqual(result["size"], result["warm_start_size"])
                self.assertEqual(result["warm_start_size"], len(degree_greedy(problem, mode)))
                self.assertEqual(result["upper_bound"], optimum)
                self.assertEqual(result["raw_upper_bound_flag"], "finite")
                self.assertEqual(result["mip_gap"], 0.)
                self.assertEqual(result["threads"], 1)
                self.assertGreaterEqual(result["total_seconds"], result["build_seconds"])
                self.assertAlmostEqual(result["solver_time_limit_seconds"],
                                       max(.001, .5 - result["build_seconds"]))
                self.assertTrue(problem.feasible(result["selected"], mode))
                json.dumps(result, allow_nan=False)

    def test_scip_tiny_target_retains_feasible_warm_start(self):
        problem = HypergraphProblem(32, tuple((2 * i, 2 * i + 1) for i in range(15)))
        for mode in ["weak", "strong"]:
            result = solve_scip_fast(problem, mode, time_limit=1e-9)
            self.assertTrue(problem.feasible(result["selected"], mode))
            self.assertGreaterEqual(result["size"], result["warm_start_size"])
            self.assertEqual(result["warm_start_size"], 17)
            self.assertEqual(result["solver_time_limit_seconds"], .001)
            self.assertGreaterEqual(result["upper_bound"], result["size"])
            json.dumps(result, allow_nan=False)

    def test_scip_events_capture_complete_improvement_and_optimal_completion(self):
        problem = HypergraphProblem(9, (
            (0, 1, 2), (0, 1, 3), (0, 2, 6), (0, 2, 8), (0, 3, 7),
            (0, 3, 8), (0, 4, 5), (0, 5, 8), (0, 6, 8), (1, 2, 4),
            (1, 6, 7), (2, 3, 5), (2, 4, 5), (2, 4, 8), (5, 7, 8)))
        for mode in ["weak", "strong"]:
            result = solve_scip_fast(problem, mode, .5, cutoff_events=True)
            optimum = len(exact_mis(problem, mode))
            self.assertEqual(result["cutoff"]["size"], optimum)
            self.assertTrue(result["cutoff"]["optimality_proven"])
            self.assertEqual(result["cutoff"]["upper_bound"], optimum)
            self.assertFalse(result["cutoff"]["deadline_missed"])
            for event in result["incumbent_events"]:
                self.assertEqual(event["size"], len(event["selected"]))
                self.assertTrue(problem.feasible(event["selected"], mode))
            self.assertEqual(result["incumbent_events"][0]["available_seconds"], 0.)
            self.assertLessEqual(result["terminal_observed_seconds"], result["terminal_available_seconds"])
            self.assertGreaterEqual(result["event_callback_seconds"], 0.)
            if mode == "weak":
                self.assertEqual(result["warm_start_size"], 5)
                self.assertTrue(any(event["source"] == "scip_best_solution_event" and event["size"] == 6
                                    for event in result["incumbent_events"]))
            json.dumps(result, allow_nan=False)

    def test_scip_event_tiny_deadline_never_backfills_late_warm_or_final_solution(self):
        problem = HypergraphProblem(32, tuple((2 * i, 2 * i + 1) for i in range(15)))
        for mode in ["weak", "strong"]:
            result = solve_scip_fast(problem, mode, 1e-9, cutoff_events=True)
            cutoff = result["cutoff"]
            self.assertGreaterEqual(result["size"], 17)
            self.assertEqual(cutoff["selected"], [])
            self.assertEqual(cutoff["size"], 0)
            self.assertFalse(cutoff["optimality_proven"])
            self.assertEqual(cutoff["status"], "deadline_snapshot")
            self.assertEqual(cutoff["upper_bound"], problem.n)
            self.assertEqual(cutoff["bound_kind"], "trivial_node_count_bound")
            self.assertTrue(cutoff["deadline_missed"])
            self.assertTrue(cutoff["overshoot"])
            self.assertLessEqual(cutoff["available_seconds"], cutoff["deadline_seconds"])
            json.dumps(result, allow_nan=False)

    def test_cutoff_rejects_future_solutions_bounds_and_optimality_and_keeps_first_tie(self):
        incumbents = [
            {"source": "initial_empty", "selected": [], "size": 0, "available_seconds": 0.},
            {"source": "first", "selected": [0, 2], "size": 2, "available_seconds": .2},
            {"source": "tie", "selected": [1, 2], "size": 2, "available_seconds": .4},
            {"source": "future", "selected": [0, 1, 2], "size": 3, "available_seconds": .6},
        ]
        bounds = [
            {"upper_bound": 5., "bound_kind": "trivial_node_count_bound", "available_seconds": 0.},
            {"upper_bound": 4., "bound_kind": "scip_dual_bound", "available_seconds": .3},
            {"upper_bound": 3., "bound_kind": "scip_dual_bound", "available_seconds": .7},
        ]
        terminal = {"status": "optimal", "optimality_proven": True, "available_seconds": .8}
        result = _cutoff_snapshot(incumbents, bounds, terminal, .5, .8)
        self.assertEqual(result["selected"], [0, 2])
        self.assertEqual(result["source"], "first")
        self.assertEqual(result["upper_bound"], 4.)
        self.assertEqual(result["upper_bound_available_seconds"], .3)
        self.assertFalse(result["optimality_proven"])
        self.assertTrue(result["deadline_missed"])
        self.assertAlmostEqual(result["overshoot_seconds"], .3)

    def test_scip_infinity_and_nonfinite_values_have_explicit_flags(self):
        class SentinelModel:
            @staticmethod
            def isInfinity(value):
                return value >= 1e20

        model = SentinelModel()
        self.assertEqual(_json_scip_number(8., model), (8., "finite", False, None))
        self.assertEqual(_json_scip_number(1e20, model), (None, "scip_infinity", True, "positive"))
        self.assertEqual(_json_scip_number(-1e20, model), (None, "scip_infinity", True, "negative"))
        self.assertEqual(_json_scip_number(math.inf, model), (None, "ieee_infinity", True, "positive"))
        self.assertEqual(_json_scip_number(-math.inf, model), (None, "ieee_infinity", True, "negative"))
        self.assertEqual(_json_scip_number(math.nan, model), (None, "nan", False, None))

    def test_invalid_definitions_and_budgets(self):
        problem = HypergraphProblem(3, ((0, 1, 2),))
        for mode in [None, "", "unknown"]:
            with self.assertRaises(ValueError):
                fast_degree_greedy(problem, mode)
            with self.assertRaises(ValueError):
                random_fast(problem, mode, 0)
            with self.assertRaises(ValueError):
                solve_scip_fast(problem, mode)
        for budget in [0., -1., math.inf, -math.inf, math.nan]:
            with self.assertRaises(ValueError):
                random_fast(problem, "weak", 0, budget)
            with self.assertRaises(ValueError):
                solve_scip_fast(problem, "weak", budget)


if __name__ == "__main__":
    unittest.main()
