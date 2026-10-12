import unittest
from mis.problem import HypergraphProblem, exact_mis
from mis.reference import solve_highs


class ReferenceTests(unittest.TestCase):
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
