"""Strict, auditable import of the public upstream hypergraph edge lists.

No optional header is guessed from the number of tokens on the first line.
This module has no machine-learning or network dependencies.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path

from .problem import HypergraphProblem


UPSTREAM_COMMIT = "708c10a95a50fb2b06dd35bf1604e65a6ec5ebe5"
UPSTREAM_REPOSITORY = (
    "https://github.com/Classpi/"
    "BIPNN-Learning-to-Solve-Binary-Integer-Programming-via-Hypergraph-Neural-Networks"
)
PAPER_URL = (
    "https://papers.nips.cc/paper_files/paper/2025/file/"
    "8693ee1ea821666f8569228d1ab38baf-Paper-Conference.pdf"
)
UPSTREAM_DATASETS = {
    "cora": {
        "filename": "cora_hypergraph.txt",
        "header": "none",
        "expected_n": 1330,
        "lf_sha256": "8b6632b64bd139eb2e66a20904e3acd1c7202ac214272271f182688c1e893be5",
        "paper_statistics": {"nodes": 1330, "edges": 1503, "incidences": 4599},
        "original_collection_identity": "unknown; upstream-labelled Cora variant",
    },
    "high": {
        "filename": "high_hypergraph.txt",
        "header": "edges_nodes",
        "expected_n": 327,
        "lf_sha256": "81de2ca7ce7f4d00b33403241ff048359b8c8040af4fcd2697b79a481a279e60",
        "paper_statistics": {"nodes": 327, "edges": 7818, "incidences": 18192},
        "original_collection_identity": "unknown; upstream-labelled High variant",
    },
    "primary": {
        "filename": "primary_hypergraph.txt",
        "header": "edges_nodes",
        "expected_n": 242,
        "lf_sha256": "9cdbe0fdc2638b8ad2357b9cbe608b2219d2119535d063d14f5cdac49e752eb4",
        "paper_statistics": {"nodes": 242, "edges": 12704, "incidences": 30729},
        "original_collection_identity": "unknown; upstream-labelled Primary variant",
    },
}


@dataclass(frozen=True)
class ImportedDataset:
    problem: HypergraphProblem
    metadata: dict

    def to_dict(self):
        """JSON input accepted by HypergraphProblem.from_json, plus its audit."""
        return {
            "n": self.problem.n,
            "edges": [list(e) for e in self.problem.edges],
            "import_metadata": self.metadata,
        }


def load_edge_list(path, *, header, node_labels=None, expected_n=None,
                   expected_sha256=None, expected_lf_sha256=None, source=None):
    """Import integer-labelled, unweighted hyperedges with an explicit policy.

    ``header='edges_nodes'`` means that the first nonblank, noncomment record
    declares the number of subsequent edge records and the vertex count.
    ``header='none'`` preserves that record as an edge, even if it has two nodes.
    Duplicate edges are removed and counted; repeated vertices within an edge
    are rejected rather than treated as loops or silently converted to sets.

    A complete ``node_labels`` sequence is required to represent unseen isolates.
    A larger declared count alone is insufficient to invent their original IDs.
    Without a declared universe, the instance covers observed vertices only and
    possible isolates outside this universe remain explicitly unknown.
    """
    if header not in ("none", "edges_nodes"):
        raise ValueError("header must explicitly be 'none' or 'edges_nodes'")
    if expected_n is not None and (type(expected_n) is not int or expected_n < 1):
        raise ValueError("expected_n must be a positive integer")
    path = Path(path)
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    lf_sha256 = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError("Raw file SHA-256 differs from the pinned dataset")
    if expected_lf_sha256 is not None and lf_sha256 != expected_lf_sha256:
        raise ValueError("LF-normalized file SHA-256 differs from the pinned dataset")
    records = [(i, line.split()) for i, line in
               enumerate(raw.decode("utf-8-sig").splitlines(), 1)
               if line.strip() and not line.lstrip().startswith("#")]
    declared_edges = declared_nodes = None
    if header == "edges_nodes":
        if not records or len(records[0][1]) != 2:
            raise ValueError("The edges_nodes header must contain exactly two integers")
        try:
            declared_edges, declared_nodes = map(int, records[0][1])
        except ValueError as exc:
            raise ValueError("The edges_nodes header must contain exactly two integers") from exc
        if declared_edges < 0 or declared_nodes < 1:
            raise ValueError("Header counts must have edges >= 0 and nodes >= 1")
        records = records[1:]
        if len(records) != declared_edges:
            raise ValueError("Header edge count does not match the number of edge records")
    parsed = []
    for line_number, fields in records:
        try:
            edge = tuple(int(v) for v in fields)
        except ValueError as exc:
            raise ValueError(f"Noninteger vertex label on line {line_number}") from exc
        if len(edge) != len(set(edge)):
            raise ValueError(f"Repeated vertices within an edge on line {line_number}")
        parsed.append(tuple(sorted(edge)))
    observed = {v for e in parsed for v in e}
    if node_labels is None:
        labels = sorted(observed)
        universe_source = "header_checked_against_observed_labels" if declared_nodes else "observed_edge_labels_only"
    else:
        labels = list(node_labels)
        if any(type(v) is not int for v in labels) or len(labels) != len(set(labels)):
            raise ValueError("node_labels must contain distinct integer vertex labels")
        labels.sort()
        if not observed.issubset(labels):
            raise ValueError("An observed vertex is absent from the supplied complete node_labels")
        universe_source = "explicit_complete_node_labels"
    if not labels:
        raise ValueError("No vertices can be inferred; supply complete node_labels")
    if declared_nodes is not None and len(labels) != declared_nodes:
        raise ValueError("Header node count differs from observed/supplied labels; unseen isolate IDs cannot be inferred")
    if expected_n is not None and len(labels) != expected_n:
        raise ValueError("Expected node count differs from observed/supplied labels; unseen isolate IDs cannot be inferred")
    id_map = {label: i for i, label in enumerate(labels)}
    unique = sorted(set(parsed))
    problem = HypergraphProblem(len(labels), tuple(tuple(id_map[v] for v in e) for e in unique))
    degrees = Counter(v for e in problem.edges for v in e)
    degree_values = [degrees[v] for v in range(problem.n)]
    pairs = {e for e in problem.edges if len(e) == 2}
    higher = [e for e in problem.edges if len(e) > 2]
    clique_pairs = {p for e in problem.edges for p in combinations(e, 2)}
    isolates = [v for v in range(problem.n) if degrees[v] == 0]
    metadata = {
        "schema_version": 1,
        "raw_filename": path.name,
        "raw_sha256": sha256,
        "lf_normalized_sha256": lf_sha256,
        "lf_hash_policy": "replace CRLF with LF only; all other bytes preserved",
        "raw_crlf_count": raw.count(b"\r\n"),
        "raw_bytes": len(raw),
        "source": source,
        "header_policy": header,
        "declared_edge_records": declared_edges,
        "declared_nodes": declared_nodes,
        "expected_nodes_check": expected_n,
        "blank_lines_and_full_line_hash_comments": "ignored",
        "edge_policy": "unweighted; order within edges ignored; duplicate edges removed",
        "repeated_vertices_within_edge": "rejected, including graph self-loop ambiguity",
        "singleton_policy": "preserved; semantics are determined by weak/strong MIS definition",
        "node_count_source": universe_source,
        "unobserved_isolates_outside_universe": "not_applicable_to_declared_universe" if declared_nodes or node_labels is not None else "unknown",
        "id_mapping_policy": "sort original integer labels ascending; map to zero-based indices",
        "original_vertex_labels": labels,
        "original_to_index": {str(label): i for label, i in id_map.items()},
        "raw_edge_records": len(parsed),
        "unique_edges": len(unique),
        "duplicate_edge_records_removed": len(parsed) - len(unique),
        "observed_nodes": len(observed),
        "nodes": problem.n,
        "raw_incidences": sum(map(len, parsed)),
        "unique_incidences": sum(map(len, unique)),
        "edge_size_histogram": dict(sorted(Counter(map(len, unique)).items())),
        "incidence_degrees": degree_values,
        "degree_min": min(degree_values),
        "degree_max": max(degree_values),
        "degree_mean": sum(degree_values) / problem.n,
        "isolated_vertex_indices": isolates,
        "isolated_original_vertex_labels": [labels[v] for v in isolates],
        "higher_order_edges": len(higher),
        "higher_order_edges_containing_existing_pair": sum(
            any(p in pairs for p in combinations(e, 2)) for e in higher),
        "higher_order_edges_with_all_pairs_present": sum(
            all(p in pairs for p in combinations(e, 2)) for e in higher),
        "original_two_node_edges": len(pairs),
        "strong_clique_unique_pairs": len(clique_pairs),
        "strong_clique_additional_pairs": len(clique_pairs - pairs),
    }
    return ImportedDataset(problem, metadata)


def load_upstream_dataset(name, data_dir=None):
    """Load one unchanged, pinned public upstream file, retaining provenance."""
    if name not in UPSTREAM_DATASETS:
        raise ValueError("Choose an upstream dataset: cora, high, or primary")
    spec = UPSTREAM_DATASETS[name]
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "data"
    filename = spec["filename"]
    imported = load_edge_list(
        data_dir / filename, header=spec["header"], expected_n=spec["expected_n"],
        expected_lf_sha256=spec["lf_sha256"], source={
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "file_url": f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_COMMIT}/data/{filename}",
            "original_collection_identity": spec["original_collection_identity"],
        })
    observed_stats = {"nodes": imported.problem.n, "edges": len(imported.problem.edges),
                      "incidences": imported.metadata["unique_incidences"]}
    paper_stats = spec["paper_statistics"]
    imported.metadata.update({
        "dataset_id": f"bipnn-upstream-{name}-{UPSTREAM_COMMIT[:7]}",
        "paper_statistics_source": {"url": PAPER_URL, "location": "Appendix F.1, Table 4, PDF page 15"},
        "paper_statistics": dict(paper_stats),
        "matches_paper_summary_statistics": observed_stats == paper_stats,
        "paper_statistics_differences": {
            key: {"file": observed_stats[key], "paper": paper_stats[key]}
            for key in observed_stats if observed_stats[key] != paper_stats[key]},
        "same_instance_as_paper": "unverified; matching summary statistics do not prove edge identity",
    })
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(UPSTREAM_DATASETS))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    imported = load_upstream_dataset(args.dataset, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(imported.to_dict(), output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"dataset": imported.metadata["dataset_id"], "nodes": imported.problem.n,
                      "edges": len(imported.problem.edges), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
