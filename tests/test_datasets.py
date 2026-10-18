import hashlib
from pathlib import Path
import tempfile
import unittest

from mis.datasets import load_edge_list, load_upstream_dataset
from mis.problem import HypergraphProblem


class DatasetTests(unittest.TestCase):
    def import_text(self, text, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.txt"
            path.write_text(text, encoding="utf-8")
            return load_edge_list(path, **kwargs)

    def test_no_header_keeps_first_pair_and_three_node_edge(self):
        imported = self.import_text("88 115\n88 223 460\n", header="none")
        self.assertEqual(imported.problem.n, 4)
        self.assertEqual(imported.problem.edges, ((0, 1), (0, 2, 3)))
        self.assertEqual(imported.metadata["original_vertex_labels"], [88, 115, 223, 460])
        self.assertEqual(imported.metadata["raw_edge_records"], 2)
        self.assertEqual(imported.metadata["unobserved_isolates_outside_universe"], "unknown")

    def test_explicit_header_counts_and_reversible_mapping(self):
        imported = self.import_text("2 3\n30 10\n30 20\n", header="edges_nodes")
        self.assertEqual(imported.problem.edges, ((0, 2), (1, 2)))
        self.assertEqual(imported.metadata["original_to_index"], {"10": 0, "20": 1, "30": 2})
        self.assertEqual(imported.metadata["incidence_degrees"], [1, 1, 2])
        self.assertEqual(imported.metadata["degree_mean"], 4 / 3)
        self.assertEqual(imported.metadata["isolated_vertex_indices"], [])
        with self.assertRaisesRegex(ValueError, "Header edge count"):
            self.import_text("3 3\n30 10\n30 20\n", header="edges_nodes")
        with self.assertRaisesRegex(ValueError, "Header node count"):
            self.import_text("2 4\n30 10\n30 20\n", header="edges_nodes")
        with self.assertRaisesRegex(ValueError, "Header node count"):
            self.import_text("2 2\n30 10\n30 20\n", header="edges_nodes")

    def test_duplicate_edges_counted_and_removed(self):
        imported = self.import_text("3 3\n1 2\n2 1\n2 3\n", header="edges_nodes")
        self.assertEqual(imported.problem.edges, ((0, 1), (1, 2)))
        self.assertEqual(imported.metadata["duplicate_edge_records_removed"], 1)
        self.assertEqual(imported.metadata["raw_incidences"], 6)
        self.assertEqual(imported.metadata["unique_incidences"], 4)
        self.assertEqual(imported.metadata["incidence_degrees"], [1, 2, 1])

    def test_repeated_vertices_invalid_and_singleton_preserved(self):
        for text in ["1 1\n", "0 0\n", "1 2 1\n"]:
            with self.assertRaisesRegex(ValueError, "Repeated vertices"):
                self.import_text(text, header="none")
        imported = self.import_text("10\n20 30\n", header="none")
        self.assertEqual(imported.problem.edges, ((0,), (1, 2)))
        self.assertFalse(imported.problem.feasible([0], "weak"))
        self.assertTrue(imported.problem.feasible([0], "strong"))

    def test_isolates_require_complete_label_universe(self):
        imported = self.import_text("1 3\n10 30\n", header="edges_nodes", node_labels=[30, 20, 10])
        self.assertEqual(imported.problem.n, 3)
        self.assertEqual(imported.metadata["isolated_vertex_indices"], [1])
        self.assertEqual(imported.metadata["isolated_original_vertex_labels"], [20])
        for labels in ([10, 20], [10, 10, 30]):
            with self.assertRaises(ValueError):
                self.import_text("10 30\n", header="none", node_labels=labels)
        with self.assertRaisesRegex(ValueError, "Expected node count"):
            self.import_text("10 30\n", header="none", expected_n=3)
        empty = self.import_text("0 2\n", header="edges_nodes", node_labels=[20, 10])
        self.assertEqual(empty.problem, HypergraphProblem(2, ()))
        self.assertEqual(empty.metadata["isolated_vertex_indices"], [0, 1])

    def test_no_header_guessing_invalid_header_and_hash(self):
        for policy in [None, "auto", "nodes_edges"]:
            with self.assertRaises(ValueError):
                self.import_text("1 2\n", header=policy)
        for text in ["? ?\n1 2\n", "3\n", "-1 2\n", "0 0\n"]:
            with self.assertRaises(ValueError):
                self.import_text(text, header="edges_nodes")
        with self.assertRaisesRegex(ValueError, "Noninteger"):
            self.import_text("0 one\n", header="none")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.import_text("1 2\n", header="none", expected_sha256="0" * 64)
        expected_hash = hashlib.sha256(b"1 2\n").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.txt"
            path.write_bytes(b"1 2\n")
            self.assertEqual(load_edge_list(path, header="none", expected_sha256=expected_hash).metadata["raw_sha256"], expected_hash)
            lf_imported = load_edge_list(path, header="none", expected_lf_sha256=expected_hash)
            path.write_bytes(b"1 2\r\n")
            crlf_imported = load_edge_list(path, header="none", expected_lf_sha256=expected_hash)
            self.assertEqual(lf_imported.problem, crlf_imported.problem)
            self.assertNotEqual(lf_imported.metadata["raw_sha256"], crlf_imported.metadata["raw_sha256"])
            self.assertEqual(crlf_imported.metadata["raw_crlf_count"], 1)
            with self.assertRaisesRegex(ValueError, "LF-normalized"):
                load_edge_list(path, header="none", expected_lf_sha256="0" * 64)

    def test_pair_redundancy_audit_does_not_remove_higher_edges(self):
        imported = self.import_text("1 2\n1 2 3\n3 4 5\n", header="none")
        self.assertEqual(len(imported.problem.edges), 3)
        self.assertEqual(imported.metadata["higher_order_edges"], 2)
        self.assertEqual(imported.metadata["higher_order_edges_containing_existing_pair"], 1)
        self.assertEqual(imported.metadata["higher_order_edges_with_all_pairs_present"], 0)
        self.assertEqual(imported.metadata["strong_clique_unique_pairs"], 6)

    def test_pinned_upstream_datasets_and_cora_discrepancy(self):
        for name, n, m, incidences, matches in [
            ("cora", 1330, 1413, 4370, False),
            ("high", 327, 7818, 18192, True),
            ("primary", 242, 12704, 30729, True),
        ]:
            imported = load_upstream_dataset(name)
            self.assertEqual((imported.problem.n, len(imported.problem.edges)), (n, m))
            self.assertEqual(imported.metadata["unique_incidences"], incidences)
            self.assertEqual(imported.metadata["matches_paper_summary_statistics"], matches)
            self.assertEqual(imported.metadata["duplicate_edge_records_removed"], 0)
            if name == "cora":
                self.assertIn((88, 115, 275), imported.problem.edges)
                self.assertEqual(imported.metadata["paper_statistics_differences"]["edges"], {"file": 1413, "paper": 1503})
            if name == "high":
                self.assertEqual(imported.metadata["higher_order_edges_containing_existing_pair"], imported.metadata["higher_order_edges"])


if __name__ == "__main__":
    unittest.main()
