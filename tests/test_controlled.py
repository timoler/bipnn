import random
import unittest

import torch
from mis.controlled import decode_stages, fingerprint, solve_controlled
from mis.problem import HypergraphProblem, repair_and_extend


class ControlledTests(unittest.TestCase):
    def test_stages_match_original_decoder_and_expose_initial_violations(self):
        problem = HypergraphProblem(9, ((0,), (1, 2, 3), (2, 4), (4, 5, 6)))
        rng = random.Random(21)
        for mode in ["weak", "strong"]:
            for scores in [[0.] * 9, [1.] * 9] + [[rng.random() for _ in range(9)] for _ in range(20)]:
                result = decode_stages(problem, scores, mode)
                self.assertEqual(result["selected"], repair_and_extend(problem, scores, mode))
                self.assertTrue(problem.feasible(result["repaired_selected"], mode))
                self.assertTrue(problem.feasible(result["selected"], mode))
                self.assertEqual(result["raw_size"] - result["repaired_size"], result["removed_by_repair"])
                self.assertEqual(result["size"] - result["repaired_size"], result["added_by_extension"])
                self.assertIn(8, result["selected"])

    def test_binding_changes_with_protocol_not_key_order(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(fingerprint({"seed": 1}), fingerprint({"seed": 2}))

    def test_training_keeps_first_forward_and_decoder_evidence(self):
        torch.set_num_threads(1)
        problem = HypergraphProblem(6, ((0, 1, 2), (2, 3, 4)))
        for mode in ["weak", "strong"]:
            record = solve_controlled(problem, mode, "w256_cra", 0, device="cpu", epochs=3)
            self.assertEqual(record["epochs_completed"], 3)
            self.assertTrue(problem.feasible(record["selected"], mode))
            self.assertEqual(record["initial"]["selected"],
                             repair_and_extend(problem, record["initial"]["scores"], mode))
            self.assertEqual(record["selected"], repair_and_extend(problem, record["scores"], mode))
            self.assertGreater(record["total_seconds"], 0)
            self.assertEqual(record["settings"]["effective_dropout"], 0)


if __name__ == "__main__":
    unittest.main()
