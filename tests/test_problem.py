import itertools
import random
import unittest

from mis.problem import HypergraphProblem, exact_mis, repair_and_extend, degree_greedy


class ProblemTests(unittest.TestCase):
    def test_definitions_differ(self):
        p = HypergraphProblem(3, ((0, 1, 2),))
        self.assertEqual(len(exact_mis(p, "weak")), 2)
        self.assertEqual(len(exact_mis(p, "strong")), 1)

    def test_isolates_and_singleton(self):
        p = HypergraphProblem(4, ((0,), (1, 2)))
        for mode, expected in [("weak", 2), ("strong", 3)]:
            result = exact_mis(p, mode)
            self.assertEqual(len(result), expected)
            self.assertIn(3, result)
            repaired = repair_and_extend(p, [0.8] * 4, mode)
            self.assertTrue(p.feasible(repaired, mode))
            self.assertIn(3, repaired)

    def test_no_edges(self):
        p = HypergraphProblem(5, ())
        for mode in ["weak", "strong"]:
            self.assertEqual(exact_mis(p, mode), list(range(5)))
            self.assertEqual(repair_and_extend(p, [0.1]*5, mode), list(range(5)))

    def test_exhaustive_energy_equivalence_and_repair(self):
        rng = random.Random(2026)
        for _ in range(12):
            n = rng.randint(3, 7)
            edges = tuple(tuple(rng.sample(range(n), rng.randint(1, n))) for _ in range(5))
            p = HypergraphProblem(n, edges)
            for mode in ["weak", "strong"]:
                terms = p.terms(mode)
                best = float("inf")
                minimizers = []
                for bits in itertools.product([0, 1], repeat=n):
                    selected = [i for i, bit in enumerate(bits) if bit]
                    energy = sum(c * all(bits[i] for i in edge) for edge, c in terms.items())
                    if energy < best:
                        best, minimizers = energy, [selected]
                    elif energy == best:
                        minimizers.append(selected)
                optimum = len(exact_mis(p, mode))
                self.assertEqual(best, -optimum)
                for s in minimizers:
                    self.assertTrue(p.feasible(s, mode))
                    self.assertEqual(len(s), optimum)
                repaired = repair_and_extend(p, [rng.random() for _ in range(n)], mode)
                self.assertTrue(p.feasible(repaired, mode))
                for v in set(range(n)) - set(repaired):
                    self.assertFalse(p.feasible(repaired + [v], mode))
                self.assertTrue(p.feasible(degree_greedy(p, mode), mode))

    def test_strong_clique_equivalence(self):
        p = HypergraphProblem(5, ((0, 1, 2), (2, 3), (4,)))
        pairs = {pair for e in p.edges for pair in itertools.combinations(e, 2)}
        for bits in itertools.product([0, 1], repeat=p.n):
            selected = [i for i, b in enumerate(bits) if b]
            clique_valid = not any(bits[a] and bits[b] for a, b in pairs)
            self.assertEqual(p.feasible(selected, "strong"), clique_valid)

    def test_validation(self):
        for edges in [((),), ((0, 0),), ((0, 3),)]:
            with self.assertRaises(ValueError):
                HypergraphProblem(3, edges)
        with self.assertRaises(ValueError):
            HypergraphProblem(3, ()).terms("unspecified")
        with self.assertRaises(ValueError):
            HypergraphProblem(3, ()).terms("weak", penalty=1)
        with self.assertRaises(ValueError):
            exact_mis(HypergraphProblem(23, ()), "weak")


if __name__ == "__main__":
    unittest.main()
