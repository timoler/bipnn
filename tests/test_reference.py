import unittest
from mis.problem import HypergraphProblem, exact_mis
from mis.reference import solve_highs, solve_scip


class ReferenceTests(unittest.TestCase):
    def test_scip_matches_enumeration_and_preserves_warm_start(self):
        p = HypergraphProblem(7, ((0,1,2), (2,3,4), (4,5)))
        for mode in ["weak", "strong"]:
            r = solve_scip(p, mode, time_limit=0.5)
            self.assertTrue(r["optimality_proven"])
            self.assertEqual(r["size"], len(exact_mis(p, mode)))
            self.assertGreaterEqual(r["size"], r["warm_start_size"])
    def test_matches_enumeration(self):
        for p in [HypergraphProblem(6, ((0,1,2), (2,3,4))),
                  HypergraphProblem(4, ((0,), (1,2))),
                  HypergraphProblem(4, ())]:
            for mode in ["weak", "strong"]:
                r = solve_highs(p, mode)
                self.assertTrue(r["optimality_proven"])
                self.assertEqual(r["size"], len(exact_mis(p, mode)))

    def test_larger_disjoint_pairs_with_isolates(self):
        p = HypergraphProblem(32, tuple((2*i, 2*i+1) for i in range(15)))
        for mode in ["weak", "strong"]:
            r = solve_highs(p, mode)
            self.assertEqual(r["size"], 17)
            self.assertTrue(r["optimality_proven"])
            self.assertTrue({30,31}.issubset(r["selected"]))


if __name__ == "__main__":
    unittest.main()
