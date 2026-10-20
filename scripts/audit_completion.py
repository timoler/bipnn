"""Independently audit the frozen thesis-completion experiment; never train.

The final summary is written only after the complete predeclared suite passes.
Training, fast-decoder and objective implementations are not used as oracles.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_controlled import (audit_decoder, distribution, finite_nonnegative,
    fingerprint, read_json, require, solution)
from summarize_anytime import (audit_random, audit_scip, audit_snapshot,
    degree_independent, equal, zero_candidate)


class IndependentProblem:
    """Small read-only model with independently checked MIS predicates."""

    def __init__(self, instance):
        self.n = instance["n"]
        require(type(self.n) is int and self.n > 0, "Invalid node count")
        canonical = []
        for edge in instance["edges"]:
            require(isinstance(edge, (list, tuple)) and edge, "Invalid empty edge")
            require(all(type(v) is int and 0 <= v < self.n for v in edge), "Invalid edge vertex")
            require(len(set(edge)) == len(edge), "Repeated vertex within an edge")
            canonical.append(tuple(sorted(edge)))
        require(len(canonical) == len(set(canonical)), "Duplicate input constraint")
        self.edges = tuple(sorted(canonical))
        self.incidence = [[] for _ in range(self.n)]
        for i, edge in enumerate(self.edges):
            for v in edge:
                self.incidence[v].append(i)

    def violations(self, selected, mode):
        require(mode in {"weak", "strong"}, "Definition must be explicit")
        chosen = set(selected)
        if mode == "weak":
            return [e for e in self.edges if all(v in chosen for v in e)]
        return [e for e in self.edges if sum(v in chosen for v in e) > 1]

    def feasible(self, selected, mode):
        return not self.violations(selected, mode)

    def maximal(self, selected, mode):
        if not self.feasible(selected, mode):
            return False
        chosen = set(selected)
        counts = [sum(v in chosen for v in edge) for edge in self.edges]
        caps = [len(e) - 1 if mode == "weak" else 1 for e in self.edges]
        return all(any(counts[i] >= caps[i] for i in self.incidence[v])
                   for v in range(self.n) if v not in chosen)


def independent_energy(problem, mode, scores, penalty):
    """Python scalar polynomial, including the unique-pair strong convention."""
    energy = -math.fsum(scores)
    if mode == "weak":
        terms = problem.edges
    else:
        terms = {pair for edge in problem.edges for pair in itertools.combinations(edge, 2)}
    return energy + penalty * math.fsum(math.prod(scores[v] for v in edge) for edge in terms)


def audit_scored_candidate(problem, mode, candidate, penalty, *, loss=True):
    audit_decoder(problem, mode, candidate, full_stages=True)
    require(problem.maximal(candidate["selected"], mode), "Decoded set is not maximal")
    if loss:
        coefficient = candidate["annealing_coefficient"]
        scores = candidate["scores"]
        expected = independent_energy(problem, mode, scores, penalty)
        expected += coefficient * math.fsum(1 - (2 * p - 1) ** 2 for p in scores)
        actual = candidate["loss_before_update"]
        require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-5, abs_tol=.005),
                f"Saved loss differs from independent polynomial: {actual} vs {expected}")


def minimal_weak_constraints(problem):
    """Return a canonical minimal antichain plus a proper-subedge witness."""
    kept, retained, witnesses = [], set(), {}
    for edge in sorted(problem.edges, key=lambda e: (len(e), e)):
        # These public graphs have maximum order five. Subset enumeration is
        # independent of the production overlap-counting pruning algorithm.
        if len(edge) <= 16:
            witness = next((small for size in range(1, len(edge))
                            for small in itertools.combinations(edge, size) if small in retained), None)
        else:
            chosen = set(edge)
            witness = next((small for small in kept if set(small) < chosen), None)
        if witness is None:
            kept.append(edge)
            retained.add(edge)
        else:
            witnesses[edge] = witness
    return tuple(sorted(kept)), witnesses


def enumerate_binary(problem, mode, penalty):
    """Exact diagnostic allowing lambda <= 1 without changing solver guards."""
    require(problem.n <= 16, "Diagnostic enumeration is limited to tiny graphs")
    rows = []
    for mask in range(1 << problem.n):
        scores = [(mask >> v) & 1 for v in range(problem.n)]
        selected = [v for v, value in enumerate(scores) if value]
        rows.append({"selected": selected, "size": len(selected),
                     "feasible": problem.feasible(selected, mode),
                     "energy": independent_energy(problem, mode, scores, penalty)})
    minimum = min(row["energy"] for row in rows)
    minimizers = [row for row in rows if math.isclose(row["energy"], minimum, rel_tol=0, abs_tol=1e-10)]
    optimum = max(row["size"] for row in rows if row["feasible"])
    return {"minimum_energy": minimum, "optimal_mis_size": optimum, "minimizers": minimizers,
            "all_minimizers_feasible": all(row["feasible"] for row in minimizers),
            "infeasible_minimizer_exists": any(not row["feasible"] for row in minimizers)}


def valid_sha(value, field):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), f"Invalid SHA-256 for {field}")


def mean(values):
    return st.mean(values) if values else None


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


SOURCES = {"mis/completion.py", "mis/anytime.py", "mis/controlled.py", "mis/indexed_loss.py",
    "mis/fast_decode.py", "mis/fast_baselines.py", "mis/problem.py", "mis/benchmark.py",
    "mis/datasets.py", "src/models.py", "src/train.py", "scripts/run_completion.py",
    "scripts/check_controlled_protocol.py", "scripts/run_controlled.py"}
TIMED_METHOD = "bipnn_completion_timed_control"
PINNED_REAL = {
    "cora": (1330, 1413, "8b6632b64bd139eb2e66a20904e3acd1c7202ac214272271f182688c1e893be5"),
    "high": (327, 7818, "81de2ca7ce7f4d00b33403241ff048359b8c8040af4fcd2697b79a481a279e60"),
    "primary": (242, 12704, "9cdbe0fdc2638b8ad2357b9cbe608b2219d2119535d063d14f5cdac49e752eb4"),
}


def generated_instance(n, k, density, seed):
    rng = random.Random(seed)
    edges = set()
    while len(edges) < n * density:
        edges.add(tuple(sorted(rng.sample(range(n), k))))
    return {"n": n, "edges": [list(e) for e in sorted(edges)],
            "generator": {"family": "uniform_fixed_edge_count", "k": k,
                          "density_m_over_n": density, "seed": seed}}


def real_input_checked(name, instance):
    n, m, digest = PINNED_REAL[name]
    raw = (ROOT / "data" / f"{name}_hypergraph.txt").read_bytes()
    require(hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == digest,
            f"Pinned real data changed: {name}")
    lines = [line.split() for line in raw.decode("utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if name != "cora":
        require([int(v) for v in lines.pop(0)] == [m, n], "Real input header changed")
    require(len(lines) == m, "Real input edge count changed")
    edge_labels = [tuple(int(v) for v in line) for line in lines]
    labels = sorted({v for e in edge_labels for v in e})
    require(len(labels) == n, "Real input universe changed")
    mapping = {label: i for i, label in enumerate(labels)}
    edges = sorted({tuple(sorted(mapping[v] for v in e)) for e in edge_labels})
    require(instance["n"] == n and instance["edges"] == [list(e) for e in edges],
            "Saved real input differs from independent raw-file parsing")
    metadata = instance["import_metadata"]
    require(metadata["lf_normalized_sha256"] == digest and metadata["original_vertex_labels"] == labels,
            "Saved real provenance/mapping differs")
    require(metadata["nodes"] == n and metadata["unique_edges"] == m and
            metadata["unique_incidences"] == sum(map(len, edges)), "Real metadata counts differ")
    return instance


def expected_inputs(directory, plan):
    cases, ablation = {}, []
    for spec in plan["ablation_inputs"]["synthetic"]:
        name = f"n{spec['n']}_k{spec['k']}_d{spec['density']}_s{spec['seed']}"
        expected = {"name": name, **generated_instance(spec["n"], spec["k"], spec["density"], spec["seed"])}
        actual = read_json(directory / "instances" / f"{name}.json")
        equal(actual, expected, "Saved ablation input differs from independent generator")
        ablation.extend((name, mode, actual) for mode in spec["definitions"])
    for dataset in plan["ablation_inputs"]["real"]["datasets"]:
        actual = read_json(directory / "instances" / f"{dataset}.json")
        require(actual["name"] == dataset + "-upstream", "Real dataset name changed")
        real_input_checked(dataset, actual)
        ablation.extend((dataset, mode, actual) for mode in plan["ablation_inputs"]["real"]["definitions"])
    for group in ("fixed", "timed"):
        for name, mode, instance in ablation:
            cases[f"{group}/{name}_{mode}"] = (group, name, mode, instance)
    rep = plan["replication"]
    for n, structure, seed in itertools.product(rep["sizes"], rep["structures"], rep["instance_seeds"]):
        k, density = structure["k"], structure["density"]
        name = f"n{n}_k{k}_d{density}_s{seed}"
        expected = {"name": name, **generated_instance(n, k, density, seed)}
        actual = read_json(directory / "instances" / f"{name}.json")
        equal(actual, expected, "Saved replication input differs from independent generator")
        mode = rep["definition"]
        cases[f"replication/{name}_{mode}"] = ("replication", name, mode, actual)
    return cases


def manifest_checked(directory):
    manifest = read_json(directory / "manifest.json")
    bound_keys = ("plan", "selection", "selection_sha256", "source_sha256", "environment", "inputs")
    require(manifest["binding"] == fingerprint({k: manifest[k] for k in bound_keys}), "Manifest binding differs")
    require(set(manifest["source_sha256"]) == SOURCES, "Bound source set differs from complete protocol")
    for filename, digest in manifest["source_sha256"].items():
        valid_sha(digest, filename)
        require(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest,
                f"Bound source changed: {filename}; use the recorded source snapshot")
    plan = manifest["plan"]
    equal(plan, read_json(ROOT / "experiments/completion-20261020.json"), "Bound plan differs from frozen plan")
    selection_path = ROOT / plan["selection_source"]
    require(hashlib.sha256(selection_path.read_bytes()).hexdigest() == manifest["selection_sha256"],
            "Original validation-selection file changed")
    equal(read_json(selection_path), manifest["selection"], "Selection object differs from bound source")
    require(manifest["selection"]["chosen"] == plan["locked_configs"] ==
            {"weak": "w1024_cra", "strong": "w1024_none"}, "Original locked configurations changed")
    prior = read_json(selection_path.parent / "manifest.json")
    require(prior["binding"] == fingerprint({"protocol": prior["protocol"], "source_sha256": prior["source_sha256"]}) and
            manifest["selection"]["validation_binding"] == prior["binding"] and
            manifest["selection"]["plan_sha256"] == fingerprint(prior["protocol"]["plan"]),
            "Selection does not retain the original validation/protocol binding")
    require(manifest["environment"]["cpu_threads"] == 1, "CPU thread control changed")
    require(plan["model_seeds"] == [0, 1, 2] and plan["width"] == 1024,
            "Completion seed/width protocol changed")
    return manifest


def variants_for(plan, group, mode):
    settings = plan[group]
    if group == "fixed":
        variants = [{"annealing": a, "penalty": settings["baseline_penalty"]} for a in settings["annealing"]]
        if mode == "weak":
            variants.extend({"annealing": "cra", "penalty": p} for p in settings["weak_additional_penalties"])
        return variants
    variants = [{"annealing": a, "penalty": settings["penalty"], "targets": settings["targets_seconds"]}
                for a in settings["annealing"]]
    if mode == "weak":
        variants.extend({"annealing": "cra", "penalty": settings["penalty"], "targets": [h]}
                        for h in settings["weak_compressed_horizons_seconds"])
    return variants


def identity_key(group, variant, seed):
    penalty = format(float(variant["penalty"]), "g")
    horizon = "" if group == "fixed" else "_h" + format(float(variant["targets"][-1]), "g")
    return f"{variant['annealing']}_p{penalty}{horizon}_seed{seed}"


def common_neural_settings(run, problem, mode, plan, variant):
    settings = run["settings"]
    require(run["definition"] == mode and run["seed"] in plan["model_seeds"], "Neural identity differs")
    require(settings["width"] == plan["width"] and settings["hidden_widths"] == [1024, 512], "Network width changed")
    require(settings["penalty"] == variant["penalty"] > 1 and settings["annealing"] == variant["annealing"],
            "Penalty/annealing differs from declared variant")
    require(settings["lr"] == 1e-4 and settings["weight_decay"] == 5e-8 and settings["effective_dropout"] == 0.,
            "Optimizer/dropout changed")
    require(settings["features"] == "fixed per-node uniform U[0,1)" and
            settings["output"] == "training-mode forward before parameter update of that step" and
            settings["gradient_check"] == "finite loss each step; final completed gradient check only",
            "Feature/output/checking convention changed")
    terms = (set(problem.edges) | {(v,) for v in range(problem.n)} if mode == "weak" else
             {pair for e in problem.edges for pair in itertools.combinations(e, 2)} |
             {(v,) for v in range(problem.n)})
    require(settings["objective_constraint_count"] == settings["message_constraint_count"] ==
            settings["decode_constraint_count"] == len(problem.edges), "A trained run reduced the original constraints")
    require(settings["objective_monomial_count"] == settings["message_monomial_count"] == len(terms) and
            settings["objective_max_order"] == max(map(len, terms)), "Objective/message support changed")
    require(settings["decode_problem"] == "original problem, even if objective/message constraints differ",
            "Decoding domain declaration changed")
    for field in ("initial_state_sha256", "feature_sha256"):
        valid_sha(run[field], field)
        require(run[field] == settings[field], "Initial hash differs between run and settings")
    total = run["run_total_seconds"]
    finite_nonnegative(total, "neural run actual total")
    require(run["total_seconds"] == total, "Neural actual total fields disagree")
    components = ("build_seconds", "training_seconds", "decode_seconds", "other_seconds")
    for field in components:
        finite_nonnegative(run[field], field)
    require(math.isclose(sum(run[k] for k in components), total, rel_tol=1e-8, abs_tol=1e-5),
            "Neural component timings do not add to actual total")
    require(run["online_decode_seconds"] == run["decode_seconds"], "Online decode time changed")
    require(type(run["epochs_completed"]) is int and run["epochs_completed"] >= 1, "Invalid completed update count")
    require(type(run["peak_allocated_bytes"]) is int and run["peak_allocated_bytes"] > 0,
            "Missing GPU peak allocation")
    return total


def audit_fixed_run(run, main, problem, mode, plan, variant):
    require(run["method"] == "bipnn_fixed_epochs_completion", "Wrong fixed-step method")
    total = common_neural_settings(run, problem, mode, plan, variant)
    settings, fixed = run["settings"], plan["fixed"]
    require(settings["fixed_epochs"] == run["epochs_completed"] == fixed["epochs"] and
            settings["checkpoint_every"] == fixed["checkpoint_every"], "Fixed step count/checkpoints changed")
    expected_schedule = "-1+1.6*(epoch-1)/(epochs-1); first=-1 if epochs=1" if variant["annealing"] == "cra" else "0"
    require(settings["schedule"] == expected_schedule and
            settings["selection"] == "first maximum decoded candidate at the declared fixed step checkpoints",
            "Fixed schedule/selection differs")
    candidates = run["candidates"]
    epochs = sorted({1, fixed["epochs"]} | set(range(fixed["checkpoint_every"], fixed["epochs"] + 1, fixed["checkpoint_every"])))
    require([c["epoch"] for c in candidates] == epochs, "Missing/spurious fixed epoch candidate")
    previous = run["build_seconds"]
    for i, candidate in enumerate(candidates):
        available = candidate["available_seconds"]
        finite_nonnegative(available, "fixed candidate availability")
        require(previous <= available <= total, "Invalid fixed candidate timeline")
        progress = (candidate["epoch"] - 1) / (fixed["epochs"] - 1)
        gamma = -1 + 1.6 * progress if variant["annealing"] == "cra" else 0.
        require(candidate["coefficient_progress"] == progress and
                candidate["gamma"] == candidate["annealing_coefficient"] == gamma, "Fixed CRA coefficient differs")
        role = "untrained_first_forward" if i == 0 else "final_epoch" if i == len(candidates) - 1 else "epoch_checkpoint"
        require(candidate["role"] == role, "Fixed candidate role changed")
        audit_scored_candidate(problem, mode, candidate, variant["penalty"])
        previous = available
    best = max(range(len(candidates)), key=lambda i: candidates[i]["size"])
    require(run["initial_index"] == 0 and run["best_index"] == best and
            run["latest_index"] == run["last_index"] == len(candidates) - 1, "Fixed candidate selection changed")
    for field, index in (("initial", 0), ("best", best), ("latest", len(candidates) - 1)):
        equal(run[field], candidates[index], f"Fixed {field} differs from its candidate")
    expected = {**candidates[best], "method": "completion_fixed_epochs", "seed": run["seed"],
        "total_seconds": total, "epochs_completed": fixed["epochs"],
        "selection_policy": "first_maximum_epoch_candidate", "variant": variant, "run_key": run["run_key"]}
    equal(main, [expected], "Fixed primary record is not its first maximum epoch candidate")
    return [c["size"] for c in candidates]


def sampling_due(targets, interval):
    horizon = targets[-1]
    return sorted({i * interval for i in range(1, math.ceil(horizon / interval))} |
                  {t - min(.1, t / 10) for t in targets})


def before_cutoff_calendar(targets, interval, cutoff):
    return {"candidate_due_seconds": [point for point in sampling_due(targets, interval) if point < cutoff],
            "earlier_snapshot_targets_seconds": [point for point in targets if point < cutoff]}


def audit_timed_run(run, main, problem, mode, plan, variant):
    require(run["method"] == TIMED_METHOD, "Wrong timed neural method")
    total = common_neural_settings(run, problem, mode, plan, variant)
    settings, targets = run["settings"], [float(t) for t in variant["targets"]]
    horizon = targets[-1]
    require(total >= horizon and run["config"] == f"w{plan['width']}_{variant['annealing']}", "Timed horizon/config differs")
    require(settings["targets_seconds"] == targets and settings["horizon_seconds"] == horizon and
            settings["candidate_interval_seconds"] == plan["candidate_interval_seconds"] and
            settings["pre_cutoff_margin_seconds"] == .1, "Timed targets/sampling changed")
    require(settings["schedule"] == ("-1+1.6*min(1,elapsed/horizon)" if variant["annealing"] == "cra" else "0") and
            settings["selection"] == "first maximum feasible candidate completed by cutoff", "Timed schedule/selection changed")
    candidates = run["candidates"]
    require(len(candidates) >= 3 and run["initial_index"] == 1 and run["last_index"] == len(candidates) - 1,
            "Missing initial/terminal timed evidence")
    zero_candidate(candidates[0], problem, mode)
    require(candidates[0]["role"] == "empty_initial_incumbent" and candidates[0]["feasible"] is True,
            "Empty incumbent role/feasibility changed")
    previous_time, previous_epoch, previous_coefficient_time = 0., 0, 0.
    due, due_index = sampling_due(targets, plan["candidate_interval_seconds"]), 0
    for i, candidate in enumerate(candidates[1:], 1):
        available = candidate["available_seconds"]
        elapsed = candidate["coefficient_elapsed_seconds"]
        epoch = candidate["epoch"]
        finite_nonnegative(available, "timed candidate availability")
        finite_nonnegative(elapsed, "coefficient elapsed time")
        require(previous_time <= available <= total and previous_coefficient_time <= elapsed <= available and
                elapsed >= run["build_seconds"], "Invalid timed/coefficient chronology")
        require(type(epoch) is int and previous_epoch <= epoch <= run["epochs_completed"], "Invalid timed candidate epoch")
        gamma = -1 + 1.6 * min(1., elapsed / horizon) if variant["annealing"] == "cra" else 0.
        require(candidate["gamma"] == candidate["annealing_coefficient"] == gamma, "Timed CRA coefficient does not match recorded elapsed time")
        role = "untrained_first_forward" if i == 1 else "terminal_diagnostic" if i == len(candidates) - 1 else "periodic_training_forward"
        require(candidate["role"] == role, "Timed candidate role changed")
        if role != "terminal_diagnostic":
            require(epoch > previous_epoch and elapsed >= previous_time,
                    "Nonterminal timed candidates reuse an old step or coefficient time")
            if role == "periodic_training_forward":
                require(due_index < len(due) and available >= due[due_index], "Candidate precedes its nominal sampling trigger")
            while due_index < len(due) and due[due_index] <= available:
                due_index += 1
        audit_scored_candidate(problem, mode, candidate, variant["penalty"])
        previous_time, previous_epoch, previous_coefficient_time = available, epoch, elapsed
    require(candidates[1]["epoch"] == 1 and candidates[-1]["epoch"] == run["epochs_completed"] and
            candidates[-1]["available_seconds"] > horizon, "Incorrect untrained/terminal candidate timing")
    require(len(run["records"]) == len(targets), "Missing timed snapshots")
    sizes = []
    for record, target in zip(run["records"], targets):
        require(record["method"] == TIMED_METHOD and record["seed"] == run["seed"] and
                record["config"] == run["config"] and record["run_key"] == run["run_key"], "Timed snapshot identity differs")
        audit_snapshot(record, candidates, target, total)
        require(type(record["epochs_completed_at_snapshot"]) is int and
                candidates[record["latest_candidate_index"]].get("epoch", 0) <= record["epochs_completed_at_snapshot"] <= run["epochs_completed"],
                "Invalid emitted-time update count")
        sizes.append(record["size"])
    require(sizes == sorted(sizes), "Timed prefix-best curve decreased")
    final = run["records"][-1]
    require(run["best_index"] == final["best_candidate_index"] and
            run["latest_index"] == final["latest_candidate_index"], "Final timed selected indices differ")
    for field, index in (("initial", 1), ("best", run["best_index"]),
                         ("latest", run["latest_index"]), ("terminal", len(candidates) - 1)):
        equal(run[field], candidates[index], f"Timed {field} differs from its candidate")
    equal(main, run["records"], "Timed primary snapshots were spliced or changed")
    return [c["size"] for c in candidates]


def audit_initial_pair(registry, instance, mode, run):
    key = (fingerprint(instance), mode, run["seed"])
    initial = run["candidates"][run["initial_index"]]
    value = {"state": run["initial_state_sha256"], "features": run["feature_sha256"],
             "scores": initial["scores"], "run_key": run["run_key"], "runs": 1}
    if key in registry:
        previous = registry[key]
        require(previous["state"] == value["state"] and previous["features"] == value["features"],
                f"Paired initialization differs: {previous['run_key']} vs {run['run_key']}")
        require(len(previous["scores"]) == len(value["scores"]) and
                all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                    for a, b in zip(previous["scores"], value["scores"])),
                "First pre-update network scores differ beyond sparse floating-point tolerance")
        previous["runs"] += 1
    else:
        registry[key] = value


def audit_case(path, manifest, *, complete=True, initial_registry=None):
    report = read_json(path)
    group, mode = report["group"], report["definition"]
    require(group in {"fixed", "timed", "replication"} and mode in {"weak", "strong"}, "Invalid case identity")
    require(Path(path).parent.name == group and Path(path).stem.endswith("_" + mode),
            "Case group/definition differs from its saved path")
    case_id = f"{group}/{Path(path).stem}"
    require(report["manifest_binding"] == manifest["binding"] and case_id in manifest["inputs"] and
            manifest["inputs"][case_id] == fingerprint(report["instance"]), "Case/input binding differs")
    equal(report["environment"], manifest["environment"], "Case environment differs")
    instance = report["instance"]
    if "generator" in instance:
        g = instance["generator"]
        expected = {"name": instance["name"], **generated_instance(instance["n"], g["k"], g["density_m_over_n"], g["seed"])}
        equal(instance, expected, "Case graph differs from independent seeded generation")
    else:
        real_input_checked(instance["name"].removesuffix("-upstream"), instance)
    problem = IndependentProblem(instance)
    plan, runs, records = manifest["plan"], report["runs"], report["records"]
    for record in records:
        solution(problem, mode, record)
        require(record["feasible"] is True, "Primary feasibility flag changed")
        finite_nonnegative(record["total_seconds"], "primary actual time")
    keys = [run["run_key"] for run in runs]
    require(len(keys) == len(set(keys)), "Duplicate complete run key")
    if initial_registry is None:
        initial_registry = {}
    if group in {"fixed", "timed"}:
        variants = {identity_key(group, variant, seed): variant
                    for seed in plan["model_seeds"] for variant in variants_for(plan, group, mode)}
        require((set(keys) == set(variants) if complete else bool(keys) and set(keys) <= set(variants)),
                "Missing or unexpected declared ablation run")
        expected_records, feasible_sizes = [], []
        for run in runs:
            key = run["run_key"]
            variant = variants[key]
            equal(run["variant"], variant, "Saved run variant differs from frozen plan")
            require(key == identity_key(group, variant, run["seed"]), "Variant/seed run key differs")
            main = [record for record in records if record.get("run_key") == key]
            if group == "fixed":
                feasible_sizes.extend(audit_fixed_run(run, main, problem, mode, plan, variant))
            else:
                feasible_sizes.extend(audit_timed_run(run, main, problem, mode, plan, variant))
            audit_initial_pair(initial_registry, instance, mode, run)
            expected_records.extend(main)
        require(len(records) == len(expected_records), "Extra/unbound primary ablation records")
    else:
        require(mode == plan["replication"]["definition"] == "weak", "Replication definition differs")
        expected_keys = {"scip_30"} | {f"{method}_{seed}" for method in ("neural", "random") for seed in plan["model_seeds"]}
        require((set(keys) == expected_keys if complete else bool(keys) and set(keys) <= expected_keys),
                "Missing or unexpected replication run")
        greedy = [record for record in records if record["method"] == "degree_greedy"]
        degree = degree_independent(problem, mode)
        require(len(greedy) == 1 and greedy[0]["selected"] == degree and greedy[0]["feasible"] is True,
                "Replication degree greedy differs")
        solution(problem, mode, greedy[0])
        require(problem.maximal(degree, mode), "Degree greedy set is not maximal")
        finite_nonnegative(greedy[0]["total_seconds"], "degree greedy elapsed")
        expected_records, feasible_sizes, bounds = list(greedy), [len(degree)], []
        target = plan["replication"]["seconds"]
        for run in runs:
            key = run["run_key"]
            main = [record for record in records if record.get("run_key") == key]
            if key.startswith("neural_"):
                require(key == f"neural_{run['seed']}", "Wrong replication model seed/key")
                variant = {"annealing": plan["replication"]["neural_annealing"],
                           "penalty": plan["replication"]["penalty"], "targets": [target]}
                require("variant" not in run, "Replication run has an undeclared ablation variant")
                feasible_sizes.extend(audit_timed_run(run, main, problem, mode, plan, variant))
                audit_initial_pair(initial_registry, instance, mode, run)
            elif key.startswith("random_"):
                require(run["seed"] in plan["model_seeds"] and key == f"random_{run['seed']}", "Wrong random seed/key")
                feasible_sizes.extend(audit_random(run, problem, mode, {"checkpoints_seconds": [target]}))
                require(run["candidates"][0]["feasible"] is True, "Random empty incumbent feasibility changed")
                equal(main, [run["single"], *run["records"]], "Random records differ from their whole trajectory")
                for candidate in [run["single"], *run["candidates"][1:]]:
                    require(problem.maximal(candidate["selected"], mode), "Random decoded candidate is not maximal")
            else:
                require(key == "scip_30" and len(main) == 1, "Wrong SCIP replication record")
                sizes, upper = audit_scip(run, main[0], problem, mode, target, degree)
                require(run["feasible"] is True and all(event["feasible"] is True for event in run["incumbent_events"]),
                        "SCIP incumbent feasibility flag changed")
                feasible_sizes.extend(sizes)
                bounds.extend(upper)
            expected_records.extend(main)
        require(len(records) == len(expected_records), "Extra/unbound replication records")
        require(all(max(feasible_sizes) <= upper + 1e-5 for upper in bounds),
                "A SCIP bound contradicts a known feasible candidate, including late diagnostics")
        if complete:
            require(len(records) == 11, "Replication record count differs")
    report["case_id"] = case_id
    return report


def audit_structural(directory, manifest, *, require_saved_inputs=True):
    data = read_json(directory / "structural.json")
    require(data["manifest_binding"] == manifest["binding"], "Structural report binding differs")
    plan = manifest["plan"]["structural"]
    items = data["weak_redundancy"]
    require([item["dataset"] for item in items] == plan["datasets"], "Missing or unexpected structural dataset")
    rows = []
    for item in items:
        name = item["dataset"]
        original = item["original_instance"]
        real_input_checked(name, original)
        trained_path = directory / "instances" / f"{name}.json"
        if require_saved_inputs or trained_path.exists():
            trained = read_json(trained_path)
            equal(original, {k: v for k, v in trained.items() if k != "name"}, "Structural and trained source graph differ")
        problem = IndependentProblem(original)
        minimal, _ = minimal_weak_constraints(problem)
        reduced = IndependentProblem(item["reduced_instance"])
        require(reduced.n == problem.n and reduced.edges == minimal, "Reduced constraints are not the full minimal antichain")
        certificate = item["certificate"]
        require(certificate["definition"] == "weak" and certificate["n"] == problem.n and
                certificate["edge_index_base"] == 0, "Structural definition/universe/index base changed")
        kept = [i for i, edge in enumerate(problem.edges) if edge in set(minimal)]
        removed = sorted(set(range(len(problem.edges))) - set(kept))
        require(certificate["kept_original_indices"] == kept and certificate["removed_original_indices"] == removed and
                certificate["original_edge_count"] == len(problem.edges) and
                certificate["retained_edge_count"] == len(minimal) and
                certificate["removed_edge_count"] == len(removed), "Structural partition/counts differ")
        require(certificate["boolean_weak_feasible_set_equivalent"] is True and
                certificate["continuous_energy_equivalent"] is False, "Structural equivalence scope changed")
        witnesses = certificate["removed_edges"]
        require([w["original_index"] for w in witnesses] == removed, "Missing/duplicated subset certificate")
        expected_map, expected_implied = [None] * len(problem.edges), [None] * len(problem.edges)
        for index, original_index in enumerate(kept):
            expected_map[original_index] = expected_implied[original_index] = index
        for witness in witnesses:
            index, kept_index = witness["original_index"], witness["witness_kept_index"]
            require(type(kept_index) is int and 0 <= kept_index < len(minimal), "Invalid retained witness index")
            require(witness["edge"] == list(problem.edges[index]) and
                    witness["witness_edge"] == list(minimal[kept_index]) and
                    set(minimal[kept_index]) < set(problem.edges[index]), "Invalid proper-subedge witness")
            expected_implied[index] = kept_index
        require(certificate["original_to_pruned"] == expected_map and
                certificate["implied_by_pruned_index"] == expected_implied, "Structural recovery maps differ")
        finite_nonnegative(certificate["incidence_visits"], "pruning work counter")
        rows.append({"dataset": name, "n": problem.n, "original_edges": len(problem.edges),
                     "retained_edges": len(minimal), "removed_edges": len(removed),
                     "retained_order_histogram": dict(sorted(Counter(map(len, minimal)).items())),
                     "removed_order_histogram": dict(sorted(Counter(len(problem.edges[i]) for i in removed).items())),
                     "all_subset_certificates_checked": True})
    equal(data["tiny_graph"], plan["tiny_graph"], "Tiny diagnostic graph differs")
    problem = IndependentProblem(data["tiny_graph"])
    table = data["penalty_boundary"]
    require([row["penalty"] for row in table] == plan["tiny_penalties"], "Missing penalty-boundary experiment")
    for row in table:
        penalty = row["penalty"]
        assignments = row["assignments"]
        require([entry["bits"] for entry in assignments] == [list(bits) for bits in itertools.product((0, 1), repeat=problem.n)],
                "Tiny enumeration omitted or reordered assignments")
        for entry in assignments:
            bits = entry["bits"]
            selected = [v for v, chosen in enumerate(bits) if chosen]
            require(entry["size"] == len(selected) and entry["feasible"] == problem.feasible(selected, "weak") and
                    math.isclose(entry["energy"], independent_energy(problem, "weak", bits, penalty), rel_tol=0, abs_tol=1e-12),
                    "Tiny assignment energy/feasibility differs from independent enumeration")
        exact = enumerate_binary(problem, "weak", penalty)
        require(math.isclose(row["minimum_energy"], exact["minimum_energy"], rel_tol=0, abs_tol=1e-12) and
                row["all_binary_minimizers_feasible"] == exact["all_minimizers_feasible"] == (penalty > 1),
                "Penalty-boundary conclusion differs")
        minima = [entry["bits"] for entry in assignments if math.isclose(entry["energy"], exact["minimum_energy"], rel_tol=0, abs_tol=1e-12)]
        equal(row["minimizing_bits"], minima, "Tiny minimizers differ")
    return {"weak_redundancy": rows, "penalty_boundary": table,
            "training_constraint_policy": "Scored runs are separately checked to retain every original constraint",
            "scope": "Exact Boolean weak feasibility certificates; no continuous or strong equivalence claim"}


def decoder_fields(candidate, prefix=""):
    predicted = "scores" in candidate
    return {prefix + field: candidate[field] if predicted else None for field in
            ("raw_size", "raw_feasible", "repaired_size", "removed_by_repair", "added_by_extension")}


def neural_rows(report):
    group, case_id, instance = report["group"], report["case_id"], report["instance"]
    rows = []
    for run in report["runs"]:
        if run["method"] not in {"bipnn_fixed_epochs_completion", TIMED_METHOD}:
            continue
        records = [record for record in report["records"] if record.get("run_key") == run["run_key"]]
        for record in records:
            initial = run["candidates"][run["initial_index"]]
            latest = run["candidates"][run["latest_index"] if group == "fixed" else record["latest_candidate_index"]]
            cutoff = None if group == "fixed" else record["cutoff_seconds"]
            initial_available = cutoff is None or initial["available_seconds"] <= cutoff
            rows.append({"group": group, "case_id": case_id, "name": instance["name"],
                "definition": report["definition"], "n": instance["n"], "m": len(instance["edges"]),
                "generator": instance.get("generator"), "scope": "synthetic_seen" if "generator" in instance and group != "replication" else
                    "synthetic_new" if group == "replication" else "real_seen",
                "trajectory_id": f"{case_id}/{run['run_key']}", "run_key": run["run_key"],
                "seed": run["seed"], "annealing": run["settings"]["annealing"],
                "penalty": run["settings"]["penalty"],
                "fixed_epochs": run["settings"].get("fixed_epochs"),
                "horizon_seconds": run["settings"].get("horizon_seconds"), "cutoff_seconds": cutoff,
                "size": record["size"], "quality_ratio": record["size"] / instance["n"],
                "latest_size": latest["size"], "initial_size": initial["size"] if initial_available else None,
                "initial_available_by_cutoff": initial_available,
                "best_minus_latest": record["size"] - latest["size"],
                "best_minus_initial": record["size"] - initial["size"] if initial_available else None,
                "selected_epoch": record.get("epoch", 0), "latest_epoch": latest.get("epoch", 0),
                "epochs_completed_at_emission": record.get("epochs_completed_at_snapshot", run["epochs_completed"]),
                "total_seconds": record["total_seconds"], "run_total_seconds": run["run_total_seconds"],
                "best_available_seconds": record.get("best_available_seconds", record["available_seconds"]),
                "overshoot_seconds": record.get("overshoot_seconds"),
                "peak_allocated_bytes": run["peak_allocated_bytes"],
                "initial_state_sha256": run["initial_state_sha256"], "feature_sha256": run["feature_sha256"],
                **decoder_fields(record), **decoder_fields(latest, "latest_")})
    return rows


def aggregate_neural(rows, group):
    chosen = [row for row in rows if row["group"] == group]
    fields = ("case_id", "annealing", "penalty", "fixed_epochs") if group == "fixed" else (
        "case_id", "annealing", "penalty", "horizon_seconds", "cutoff_seconds")
    result = []
    for key in sorted({tuple(row[field] for field in fields) for row in chosen}):
        values = sorted((row for row in chosen if tuple(row[field] for field in fields) == key), key=lambda row: row["seed"])
        require([row["seed"] for row in values] == [0, 1, 2], "Aggregate lacks the three technical model seeds")
        first = values[0]
        row = {field: first[field] for field in ("case_id", "name", "scope", "definition", "n", "m", "generator",
            "annealing", "penalty", "fixed_epochs", "horizon_seconds", "cutoff_seconds")}
        row.update({"model_seeds": [r["seed"] for r in values], "sizes": [r["size"] for r in values],
            "latest_sizes": [r["latest_size"] for r in values], "initial_sizes": [r["initial_size"] for r in values],
            "mean_size": mean([r["size"] for r in values]), "mean_quality_ratio": mean([r["quality_ratio"] for r in values]),
            "latest_mean_size": mean([r["latest_size"] for r in values]),
            "initial_mean_size": mean([r["initial_size"] for r in values if r["initial_size"] is not None]),
            "mean_best_minus_latest": mean([r["best_minus_latest"] for r in values]),
            "mean_best_minus_initial": mean([r["best_minus_initial"] for r in values if r["best_minus_initial"] is not None]),
            "initial_available_count": sum(r["initial_available_by_cutoff"] for r in values),
            "predicted_records": sum(r["raw_feasible"] is not None for r in values),
            "empty_incumbent_rate": mean([r["raw_feasible"] is None for r in values]),
            "raw_feasible_rate": mean([r["raw_feasible"] for r in values if r["raw_feasible"] is not None]),
            "latest_raw_feasible_rate": mean([r["latest_raw_feasible"] for r in values if r["latest_raw_feasible"] is not None]),
            "raw_empty_rate": mean([r["raw_size"] == 0 for r in values if r["raw_size"] is not None]),
            "raw_size_mean": mean([r["raw_size"] for r in values if r["raw_size"] is not None]),
            "repaired_size_mean": mean([r["repaired_size"] for r in values if r["repaired_size"] is not None]),
            "removed_mean": mean([r["removed_by_repair"] for r in values if r["removed_by_repair"] is not None]),
            "added_mean": mean([r["added_by_extension"] for r in values if r["added_by_extension"] is not None]),
            "actual_seconds": [r["total_seconds"] for r in values],
            "best_available_seconds": [r["best_available_seconds"] for r in values],
            "epochs_at_emission": [r["epochs_completed_at_emission"] for r in values],
            "peak_allocated_bytes": [r["peak_allocated_bytes"] for r in values]})
        result.append(row)
    return result


def paired_row(left, right, *, label, cross_group=False):
    same_case = (left["name"] == right["name"] and left["definition"] == right["definition"] and
                 left["n"] == right["n"] and left["m"] == right["m"]) if cross_group else left["case_id"] == right["case_id"]
    require(same_case and left["model_seeds"] == right["model_seeds"], "Nonpaired comparison")
    result = {field: left[field] for field in ("case_id", "name", "scope", "definition", "n", "generator", "cutoff_seconds")}
    result.update({"comparison": label, "left_case_id": left["case_id"], "right_case_id": right["case_id"],
        "left_sizes": left["sizes"], "right_sizes": right["sizes"],
        "left_mean_size": left["mean_size"], "right_mean_size": right["mean_size"],
        "paired_seed_differences": [a - b for a, b in zip(left["sizes"], right["sizes"])],
        "mean_size_difference": left["mean_size"] - right["mean_size"],
        "percent_mean_quality_difference": 100 * (left["mean_size"] / right["mean_size"] - 1) if right["mean_size"] else None})
    return result


def aggregate_comparisons(rows, fields=("definition",)):
    groups = []
    for key in sorted({tuple(row[field] for field in fields) for row in rows}):
        selected = [row for row in rows if tuple(row[field] for field in fields) == key]
        differences = [row["mean_size_difference"] for row in selected]
        groups.append({**dict(zip(fields, key)), "instances": len(selected),
            "mean_size_difference": mean(differences),
            "mean_paired_percent": mean([row["percent_mean_quality_difference"] for row in selected
                                         if row["percent_mean_quality_difference"] is not None]),
            "wins_ties_losses": [sum(function(value) for value in differences) for function in
                (lambda value: value > 1e-10, lambda value: abs(value) <= 1e-10, lambda value: value < -1e-10)]})
    return groups


def comparisons(fixed, timed):
    fixed_pairs, timed_pairs, compressed, penalties = [], [], [], []
    for cra in [row for row in fixed if row["annealing"] == "cra" and row["penalty"] == 2]:
        none = next(row for row in fixed if row["case_id"] == cra["case_id"] and row["annealing"] == "none")
        fixed_pairs.append(paired_row(cra, none, label="fixed1600 CRA minus none"))
    for cra in [row for row in timed if row["annealing"] == "cra" and row["horizon_seconds"] == 30]:
        none = next(row for row in timed if row["case_id"] == cra["case_id"] and row["annealing"] == "none" and
                    row["cutoff_seconds"] == cra["cutoff_seconds"])
        timed_pairs.append(paired_row(cra, none, label="fixed30 wall-clock CRA minus none"))
    for short in [row for row in timed if row["annealing"] == "cra" and row["horizon_seconds"] < 30]:
        long = next(row for row in timed if row["case_id"] == short["case_id"] and row["annealing"] == "cra" and
                    row["horizon_seconds"] == 30 and row["cutoff_seconds"] == short["cutoff_seconds"])
        compressed.append(paired_row(short, long, label="compressed CRA minus fixed30 CRA at same cutoff"))
    for row in [row for row in fixed if row["definition"] == "weak" and row["annealing"] == "cra"]:
        baseline = next(base for base in fixed if base["case_id"] == row["case_id"] and base["annealing"] == "cra" and base["penalty"] == 2)
        penalties.append({**paired_row(row, baseline, label="fixed1600 lambda sensitivity relative to prespecified lambda2"), "penalty": row["penalty"]})
    return {"fixed_cra_comparisons": {"by_case": fixed_pairs, "by_definition": aggregate_comparisons(fixed_pairs)},
            "timed_cra_comparisons": {"by_case": timed_pairs, "by_definition_cutoff": aggregate_comparisons(timed_pairs, ("definition", "cutoff_seconds"))},
            "compressed_comparisons": {"by_case": compressed, "by_cutoff": aggregate_comparisons(compressed, ("cutoff_seconds",))},
            "penalty_comparisons": {"by_case": penalties, "by_penalty": aggregate_comparisons(penalties, ("penalty",))}}


def replication_rows(reports, all_neural):
    result = []
    for report in reports:
        if report["group"] != "replication":
            continue
        records, instance = report["records"], report["instance"]
        neural = sorted((r for r in all_neural if r["case_id"] == report["case_id"]), key=lambda r: r["seed"])
        randoms = sorted((r for r in records if r["method"] == "random_restart_same_decoder"), key=lambda r: r["seed"])
        singles = sorted((r for r in records if r["method"] == "random_score_same_decoder"), key=lambda r: r["seed"])
        scip = next(r for r in records if r["method"] == "scip_milp_greedy_start")
        degree = next(r for r in records if r["method"] == "degree_greedy")
        methods = {"neural": mean([r["size"] for r in neural]), "random_restart": mean([r["size"] for r in randoms]),
                   "random_single": mean([r["size"] for r in singles]), "scip": scip["size"], "degree": degree["size"]}
        row = {"case_id": report["case_id"], "name": instance["name"], "definition": report["definition"],
            "n": instance["n"], "m": len(instance["edges"]), "generator": instance["generator"],
            "model_seeds": [r["seed"] for r in neural], "neural_sizes": [r["size"] for r in neural],
            "random_sizes": [r["size"] for r in randoms], "random_single_sizes": [r["size"] for r in singles],
            "neural_mean_size": methods["neural"], "random_mean_size": methods["random_restart"],
            "scip_size": scip["size"], "degree_size": degree["size"], "method_mean_sizes": methods,
            "method_mean_ratios": {name: value / instance["n"] for name, value in methods.items()},
            "paired_percent": {name: 100 * (methods["neural"] / value - 1) if value else None
                               for name, value in methods.items() if name != "neural"},
            "scip_upper_bound": scip["upper_bound"], "scip_optimality_proven": scip["optimality_proven"],
            "scip_status": scip["status"], "scip_actual_seconds": scip["total_seconds"],
            "scip_completion_size": scip["completion_size"], "scip_completion_status": scip["completion_status"],
            "neural_actual_seconds": [r["total_seconds"] for r in neural],
            "random_actual_seconds": [r["total_seconds"] for r in randoms],
            "neural_mean_best_minus_latest": mean([r["best_minus_latest"] for r in neural])}
        result.append(row)
    return result


def replication_group(rows):
    methods = ("degree", "random_single", "random_restart", "scip")
    return {"graphs": len(rows), "model_seeds_per_graph": 3,
        "comparisons": {method: {"mean_paired_percent": mean([r["paired_percent"][method] for r in rows
                if r["paired_percent"][method] is not None]),
            "wins_ties_losses": [sum(function(r["method_mean_sizes"]["neural"] - r["method_mean_sizes"][method])
                                     for r in rows) for function in
                (lambda value: value > 1e-10, lambda value: abs(value) <= 1e-10, lambda value: value < -1e-10)]}
            for method in methods},
        "method_quality_ratios": {method: {"mean": mean([r["method_mean_ratios"][method] for r in rows]),
            "min": min(r["method_mean_ratios"][method] for r in rows), "max": max(r["method_mean_ratios"][method] for r in rows)}
            for method in ("neural", *methods)}, "scip_proven_graphs": sum(r["scip_optimality_proven"] for r in rows)}


def replication_groups(rows):
    result = []
    for key in sorted({(r["n"], r["generator"]["k"], r["generator"]["density_m_over_n"]) for r in rows}):
        chosen = [r for r in rows if (r["n"], r["generator"]["k"], r["generator"]["density_m_over_n"]) == key]
        require(sorted(r["generator"]["seed"] for r in chosen) == [20261027, 20261028, 20261029], "Replication group lacks its three graph seeds")
        result.append({"n": key[0], "k": key[1], "density_m_over_n": key[2],
                       "graph_seeds": [r["generator"]["seed"] for r in chosen], **replication_group(chosen)})
    return result


def audit_timing_ranges(reports, *, enforce=True):
    """Describe every saved run clock; reject obvious suspension-sized gaps.

    Thresholds diagnose gross interruptions, not speed or statistical outliers.
    They never select individual favorable runs for replacement.
    """
    rows, emitted, flags = [], [], []
    for report in reports:
        case_id = report["case_id"]
        for record in report["records"]:
            if "run_key" not in record:
                finite_nonnegative(record["total_seconds"], "standalone record elapsed")
                if record["total_seconds"] > 60:
                    flags.append({"case_id": case_id, "reason": "standalone record exceeds 60 seconds"})
        for run in report["runs"]:
            key = run["run_key"]
            total = run.get("run_total_seconds", run.get("total_seconds"))
            finite_nonnegative(total, "saved complete run elapsed")
            settings = run.get("settings", {})
            fixed = "fixed_epochs" in settings
            main = [record for record in report["records"] if record.get("run_key") == key]
            horizon = settings.get("horizon_seconds", run.get("time_limit_seconds"))
            if horizon is None and not fixed:
                targets = [record.get("budget_seconds") for record in main if record.get("budget_seconds") is not None]
                horizon = max(targets) if targets else None
            if "trial_log" in run:
                times = [entry[0] for entry in run["trial_log"]]
                evidence = "all_random_trial_completions"
            elif "candidates" in run:
                times = [candidate["available_seconds"] for candidate in run["candidates"]]
                evidence = "network_candidate_completions"
            else:
                times = [event["available_seconds"] for field in ("incumbent_events", "bound_events") for event in run[field]]
                evidence = "solver_incumbent_and_bound_events"
            times = sorted([0., *times, total])
            require(all(math.isfinite(t) and 0 <= t <= total for t in times), "Invalid saved event time during timing audit")
            gaps = [(b - a, a, b) for a, b in zip(times, times[1:])]
            gap, begin, end = max(gaps)
            row = {"case_id": case_id, "group": report["group"], "run_key": key, "method": run["method"],
                "run_total_seconds": total, "nominal_horizon_seconds": horizon,
                "fixed_epochs": settings.get("fixed_epochs"), "max_evidence_gap_seconds": gap,
                "gap_start_seconds": begin, "gap_end_seconds": end, "evidence": evidence,
                "other_seconds": run.get("other_seconds")}
            rows.append(row)
            reasons = []
            if fixed and total > 600:
                reasons.append("fixed-step run exceeds 600-second gross-interruption diagnostic")
            if horizon is not None and total > max(60, 2 * horizon):
                reasons.append("timed run exceeds max(60 seconds, twice its horizon)")
            if gap > 60:
                reasons.append("saved completion/event gap exceeds 60 seconds")
            if run.get("other_seconds", 0) > 60:
                reasons.append("unassigned timing component exceeds 60 seconds")
            if reasons:
                flags.append({**row, "reasons": reasons})
            for record in main:
                budget = record.get("budget_seconds")
                if budget is None:
                    continue
                actual = record["total_seconds"]
                finite_nonnegative(actual, "actual emitted snapshot time")
                lateness = max(0., actual - budget)
                emitted.append({"case_id": case_id, "group": report["group"], "run_key": key,
                    "budget_seconds": budget, "emitted_seconds": actual, "lateness_seconds": lateness})
                if actual > max(60, 2 * budget):
                    flags.append({"case_id": case_id, "run_key": key, "budget_seconds": budget,
                                  "reason": "snapshot emission exceeds max(60 seconds, twice its cutoff)", "actual_seconds": actual})
    if enforce:
        require(not flags, "Gross timing interruption found in included results; retain evidence and resolve before final completion: " + str(flags))
    groups = []
    for key in sorted({(row["group"], row["method"]) for row in rows}):
        chosen = [row for row in rows if (row["group"], row["method"]) == key]
        groups.append({"group": key[0], "method": key[1], "runs": len(chosen),
            "run_total_seconds": distribution([row["run_total_seconds"] for row in chosen]),
            "max_evidence_gap_seconds": max(row["max_evidence_gap_seconds"] for row in chosen)})
    return {"all_included_runs_checked": enforce, "run_count": len(rows), "timed_snapshot_count": len(emitted),
        "groups": groups, "run_rows": rows,
        "minimum_run_total_seconds": min(row["run_total_seconds"] for row in rows),
        "maximum_run_total_seconds": max(row["run_total_seconds"] for row in rows),
        "largest_saved_gap": max(rows, key=lambda row: row["max_evidence_gap_seconds"]),
        "latest_emitted_snapshot": max(emitted, key=lambda row: row["emitted_seconds"]) if emitted else None,
        "largest_lateness": max(emitted, key=lambda row: row["lateness_seconds"]) if emitted else None,
        "gross_interruption_flags": flags,
        "diagnostic_thresholds": "Timed total/emission >max(60s,2*horizon/cutoff), fixed total>600s, saved evidence gap>60s or unassigned component>60s. No automatic row-level exclusion."}


def audit_schedule_match(directory, parent_directory, parent, summary, initial_registry):
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    keys = {"parent_manifest_binding", "parent_manifest_sha256", "parent_plan_sha256", "plan", "plan_sha256",
            "source_sha256", "selection_sha256", "environment", "inputs", "input_file_sha256"}
    require(set(manifest) - {"binding", "git_commit_at_start", "created_at_utc"} == keys and
            manifest["binding"] == fingerprint({key: manifest[key] for key in keys}), "Schedule-match manifest binding differs")
    require(manifest["parent_manifest_binding"] == parent["binding"] and
            manifest["parent_manifest_sha256"] == hashlib.sha256((parent_directory / "manifest.json").read_bytes()).hexdigest(),
            "Schedule-match parent pointer/bytes differ")
    parent_plan_path = ROOT / "experiments/completion-20261020.json"
    child_plan_path = ROOT / "experiments/schedule-match-20261020.json"
    require(manifest["parent_plan_sha256"] == hashlib.sha256(parent_plan_path.read_bytes()).hexdigest() and
            manifest["plan_sha256"] == hashlib.sha256(child_plan_path.read_bytes()).hexdigest(), "Schedule-match plan bytes changed")
    plan = manifest["plan"]
    equal(plan, read_json(child_plan_path), "Bound supplementary plan differs")
    require(plan["parent_manifest_binding"] == parent["binding"] and plan["parent_plan"] == "experiments/completion-20261020.json",
            "Supplementary plan refers to a different parent")
    require(set(manifest["source_sha256"]) == SOURCES | {"scripts/run_schedule_match.py"}, "Supplement source set differs")
    for filename, digest in manifest["source_sha256"].items():
        valid_sha(digest, filename)
        require(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, f"Supplement bound source changed: {filename}")
        if filename in SOURCES:
            require(digest == parent["source_sha256"][filename], "Supplement changed a frozen parent source")
    equal(manifest["environment"], parent["environment"], "Supplement actual environment differs from parent")
    require(manifest["selection_sha256"] == parent["selection_sha256"], "Supplement changed original locked selection bytes")
    require(plan["model_seeds"] == parent["plan"]["model_seeds"] == [0, 1, 2] and
            plan["width"] == parent["plan"]["width"] == 1024 and plan["definition"] == "weak" and
            plan["candidate_interval_seconds"] == parent["plan"]["candidate_interval_seconds"] == 1 and
            plan["comparison_cutoff_seconds"] == 15, "Supplement changes unrelated settings")
    variant = {"annealing": "cra", "penalty": 2, "targets": [5, 15], "sampling_control": "matched_pre15_calendar"}
    equal(plan["variant"], variant, "Supplement variant differs")
    equal(plan["expected_counts"], {"cases": 5, "neural_runs": 15, "primary_records": 30}, "Supplement count plan differs")
    interval = plan["candidate_interval_seconds"]
    child_calendar = before_cutoff_calendar([5, 15], interval, 15)
    parent_calendar = before_cutoff_calendar(parent["plan"]["timed"]["targets_seconds"], interval, 15)
    equal(child_calendar, parent_calendar, "Pre-15 candidate/snapshot calendar is not matched")
    equal(before_cutoff_calendar([5], interval, 5), before_cutoff_calendar([5, 15, 30], interval, 5),
          "Original compressed5 contrast has a pre-cutoff calendar mismatch")
    unmatched = before_cutoff_calendar([15], interval, 15)
    require(unmatched != parent_calendar, "Supplement reason no longer matches the original15 calendar")
    names = [f"n{spec['n']}_k{spec['k']}_d{spec['density']}_s{spec['seed']}"
             for spec in parent["plan"]["ablation_inputs"]["synthetic"] if "weak" in spec["definitions"]]
    if "weak" in parent["plan"]["ablation_inputs"]["real"]["definitions"]:
        names += parent["plan"]["ablation_inputs"]["real"]["datasets"]
    require([spec["name"] for spec in plan["inputs"]] == names and len(names) == 5, "Supplement selected a different input subset")
    require(set(manifest["inputs"]) == {f"schedule_match/{name}_weak" for name in names} and
            set(manifest["input_file_sha256"]) == {f"instances/{name}.json" for name in names}, "Supplement input set differs")
    actual = {path.stem for path in (directory / "cases").glob("*.json")}
    require(actual == {name + "_weak" for name in names}, "Missing/extra supplementary case file")
    rows, runs, reports = [], [], []
    paired_before = sum(entry["runs"] for entry in initial_registry.values() if entry["runs"] > 1)
    for spec in plan["inputs"]:
        name = spec["name"]
        relative = f"instances/{name}.json"
        require(spec["parent_instance"] == relative, "Supplement instance filename differs")
        instance_path = parent_directory / relative
        require(hashlib.sha256(instance_path.read_bytes()).hexdigest() == manifest["input_file_sha256"][relative],
                "Supplement parent input bytes differ")
        instance = read_json(instance_path)
        signature = fingerprint(instance)
        require(signature == spec["fingerprint"] == manifest["inputs"][f"schedule_match/{name}_weak"] and
                signature == parent["inputs"][f"fixed/{name}_weak"] == parent["inputs"][f"timed/{name}_weak"],
                "Supplement and parent full input fingerprints differ")
        report = read_json(directory / "cases" / f"{name}_weak.json")
        require(report["group"] == "schedule_match" and report["definition"] == "weak" and
                report["manifest_binding"] == manifest["binding"] and
                report["parent_manifest_binding"] == parent["binding"], "Supplement case identity/binding differs")
        equal(report["environment"], manifest["environment"], "Supplement case environment differs")
        equal(report["instance"], instance, "Supplement case uses different instance")
        problem = IndependentProblem(instance)
        expected_keys = {f"cra_p2_h15_matched_seed{seed}" for seed in plan["model_seeds"]}
        require(len(report["runs"]) == 3 and {run["run_key"] for run in report["runs"]} == expected_keys,
                "Supplement has missing/duplicate/unexpected whole runs")
        flattened = []
        for run in report["runs"]:
            require(run["seed"] in plan["model_seeds"] and run["run_key"] == f"cra_p2_h15_matched_seed{run['seed']}",
                    "Supplement seed/run key differs")
            equal(run["variant"], variant, "Supplement run variant differs")
            main = [record for record in report["records"] if record.get("run_key") == run["run_key"]]
            audit_timed_run(run, main, problem, "weak", parent["plan"], variant)
            require((signature, "weak", run["seed"]) in initial_registry, "Supplement lacks a parent initialization pair")
            audit_initial_pair(initial_registry, instance, "weak", run)
            flattened.extend(run["records"])
            runs.append(run)
        equal(report["records"], flattened, "Supplement records were spliced/reordered relative to whole runs")
        require(len(report["records"]) == 6, "Supplement case snapshot count differs")
        report["case_id"] = f"schedule_match/{name}_weak"
        reports.append(report)
        derived = neural_rows(report)
        for row in derived:
            row.update({"targets_seconds": [5, 15], "sampling_calendar_verified": True})
        rows.extend(derived)
        print(f"AUDITED schedule_match/{name}_weak: 3 whole runs, 6 primary records", flush=True)
    grouped = aggregate_neural(rows, "schedule_match")
    require(len(rows) == 30 and len(grouped) == 10 and len(runs) == 15, "Supplement summary dimensions differ")
    require(sum(entry["runs"] for entry in initial_registry.values() if entry["runs"] > 1) == paired_before + 15,
            "Supplement initialization pairing coverage differs")
    pairs = []
    for row in grouped:
        reference = next(base for base in summary["timed_rows"] if base["name"] == row["name"] and
            base["definition"] == "weak" and base["annealing"] == "cra" and base["horizon_seconds"] == 30 and
            base["cutoff_seconds"] == row["cutoff_seconds"])
        pairs.append(paired_row(row, reference, cross_group=True,
            label="matched-calendar horizon15 minus fixed30 at same strict cutoff"))
    try:
        directory_label = directory.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        directory_label = str(directory.resolve())
    return {"complete": True, "manifest_binding": manifest["binding"], "parent_manifest_binding": parent["binding"],
        "git_commit_at_start": manifest["git_commit_at_start"],
        "directory": directory_label, "manifest_source": directory_label + "/manifest.json",
        "counts": {"case_files": 5, "neural_runs": 15, "primary_records": 30,
                   "neural_candidates_audited": sum(sum("scores" in c for c in run["candidates"]) for run in runs)},
        "neural_rows": rows, "rows": grouped, "sampling_calendar_verified": True,
        "timing_integrity": audit_timing_ranges(reports),
        "calendars": {"matched15_before15": child_calendar, "parent30_before15": parent_calendar,
                      "original_unmatched15_before15": unmatched},
        "comparisons": {"by_case": pairs, "by_cutoff": aggregate_comparisons(pairs, ("cutoff_seconds",))},
        "neural_run_profile": {field: distribution([run[field] for run in runs]) for field in
            ("run_total_seconds", "build_seconds", "training_seconds", "online_decode_seconds", "epochs_completed", "peak_allocated_bytes")},
        "registration": plan["registration"],
        "scope": "Supplement fixed before its execution following source-calendar review; no outcome selection. Actual times/epochs/response latency may differ."}


def attach_schedule_match(summary, child):
    summary["schedule_match"] = child
    primary = [row for row in summary["compressed_comparisons"]["by_case"] if row["cutoff_seconds"] == 5]
    primary += [row for row in child["comparisons"]["by_case"] if row["cutoff_seconds"] == 15]
    require(len(primary) == 10, "Primary schedule comparison coverage differs")
    diagnostic = [row for row in summary["compressed_comparisons"]["by_case"] if row["cutoff_seconds"] == 15]
    summary["primary_schedule_comparisons"] = {"by_case": primary, "by_cutoff": aggregate_comparisons(primary, ("cutoff_seconds",))}
    summary["unmatched15_diagnostics"] = {"by_case": diagnostic, "by_cutoff": aggregate_comparisons(diagnostic, ("cutoff_seconds",))}
    summary["compressed_comparisons"]["scope"] = "Frozen main contrasts; original15 is diagnostic. Use primary_schedule_comparisons for matched primary conclusions."
    summary["combined_counts"] = {"total_neural_runs": summary["counts"]["total_neural_runs"] + child["counts"]["neural_runs"],
        "primary_records": summary["counts"]["primary_records"] + child["counts"]["primary_records"],
        "case_files": summary["counts"]["case_files"] + child["counts"]["case_files"],
        "random_runs": summary["counts"]["random_runs"], "scip_runs": summary["counts"]["scip_runs"],
        "neural_candidates_audited": summary["counts"]["neural_candidates_audited"] + child["counts"]["neural_candidates_audited"]}
    require(summary["combined_counts"]["total_neural_runs"] == 222 and summary["combined_counts"]["primary_records"] == 429,
            "Combined main/supplement counts differ")
    summary["limits"].append("Original15 sampling mismatch retained as diagnostic; primary15 uses the separately bound matched-calendar supplement")
    summary["complete"] = summary["main_suite_complete"] and child["complete"]


def audit_interrupted_archive(directory, child_directory, parent, initial_registry):
    """Preserve and verify the whole excluded attempt, without using its scores."""
    evidence_path = directory / "interruption.json"
    evidence = read_json(evidence_path)
    manifest_path = directory / "manifest.json"
    archived, formal = read_json(manifest_path), read_json(child_directory / "manifest.json")
    require(evidence["event"] == "extended_execution_interruption" and evidence["included_in_primary_analysis"] is False,
            "Interrupted attempt exclusion evidence differs")
    require(evidence["original_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "Archived original manifest raw bytes differ from interruption evidence")
    bound_keys = ("parent_manifest_binding", "parent_manifest_sha256", "parent_plan_sha256", "plan", "plan_sha256",
                  "source_sha256", "selection_sha256", "environment", "inputs", "input_file_sha256")
    equal({key: archived[key] for key in bound_keys}, {key: formal[key] for key in bound_keys},
          "Formal retry did not retain the entire excluded attempt's frozen protocol/source/environment/inputs")
    require(archived["binding"] == formal["binding"] == fingerprint({key: archived[key] for key in bound_keys}),
            "Archived supplement binding differs")
    names = [spec["name"] for spec in formal["plan"]["inputs"]]
    require({path.stem for path in (directory / "cases").glob("*.json")} == {name + "_weak" for name in names},
            "Interrupted archive is not the full five-case attempt")
    registry = {key: dict(value) for key, value in initial_registry.items()}
    reports, saved_runs, candidate_count = [], [], 0
    for name in names:
        report = read_json(directory / "cases" / f"{name}_weak.json")
        require(report["group"] == "schedule_match" and report["definition"] == "weak" and
                report["manifest_binding"] == archived["binding"] and
                report["parent_manifest_binding"] == parent["binding"], "Archived case identity/binding differs")
        equal(report["environment"], archived["environment"], "Archived environment differs")
        signature = fingerprint(report["instance"])
        require(signature == archived["inputs"][f"schedule_match/{name}_weak"], "Archived full instance differs")
        problem = IndependentProblem(report["instance"])
        require(len(report["runs"]) == 3 and
                {run["run_key"] for run in report["runs"]} == {f"cra_p2_h15_matched_seed{seed}" for seed in (0, 1, 2)},
                "Interrupted archive is missing/duplicating model seeds")
        flattened = []
        for run in report["runs"]:
            require(run["run_key"] == f"cra_p2_h15_matched_seed{run['seed']}", "Archived model seed/key differs")
            equal(run["variant"], archived["plan"]["variant"], "Archived variant differs")
            main = [record for record in report["records"] if record.get("run_key") == run["run_key"]]
            audit_timed_run(run, main, problem, "weak", parent["plan"], archived["plan"]["variant"])
            require((signature, "weak", run["seed"]) in registry, "Archived attempt lacks a formal initialization match")
            audit_initial_pair(registry, report["instance"], "weak", run)
            flattened.extend(run["records"])
            saved_runs.append({"case": name + "_weak", "run_key": run["run_key"], "run_total_seconds": run["run_total_seconds"]})
            candidate_count += sum("scores" in c for c in run["candidates"])
        equal(report["records"], flattened, "Archived snapshots were spliced")
        require(len(report["records"]) == 6, "Interrupted archive lacks the full 30 snapshots")
        report["case_id"] = f"interrupted_schedule_match/{name}_weak"
        reports.append(report)
    sort_key = lambda row: (row["case"], row["run_key"])
    equal(sorted(evidence["runs"], key=sort_key), sorted(saved_runs, key=sort_key), "Interruption inventory differs from archived raw runs")
    outliers = sorted((row for row in saved_runs if row["run_total_seconds"] > 60), key=sort_key)
    equal(sorted(evidence["outliers_over_60_seconds"], key=sort_key), outliers, "Interrupted outlier inventory differs")
    require(len(saved_runs) == 15 and len(outliers) == 1, "Archived incident is not one long interruption within a full 15-run attempt")
    timing = audit_timing_ranges(reports, enforce=False)
    try:
        label = directory.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        label = str(directory.resolve())
    print("AUDITED excluded interrupted supplement: all 15 runs/30 snapshots retained; one >60s run", flush=True)
    return {"included_in_primary_analysis": False, "archive_complete_evidence_checked": True,
        "directory": label, "interruption_source": label + "/interruption.json",
        "interruption_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "manifest_binding": archived["binding"], "counts": {"case_files": 5, "neural_runs": 15, "primary_records": 30,
            "neural_candidates_audited": candidate_count},
        "outliers_over_60_seconds": outliers, "timing_integrity": timing,
        "whole_attempt_excluded": True, "all_15_runs_retried_with_identical_bound_settings": True,
        "specific_os_cause": "not established", "exclusion_scope": evidence["exclusion_scope"],
        "scope": "Entire first attempt retained and excluded irrespective of quality; formal retry, not row-level replacement."}


def fmt(value, decimals=2):
    return "—" if value is None else f"{value:.{decimals}f}"


def pct(value):
    return "未定义" if value is None else f"{value:+.2f}%"


def ratio(value):
    return "—" if value is None else f"{value:.1%}"


def report_text(summary, manifest):
    counts = summary["counts"]
    combined, child = summary["combined_counts"], summary["schedule_match"]
    overall = summary["replication_overall"]
    environment = manifest["environment"]
    lines = ["# 毕业论文实验收尾：单因素消融、新生成图复核与结构证书", "",
        "协议日期：2026-10-20。以下数字均来自完整独立核验通过的本轮结果；此前批次保持冻结。", "",
        f"本轮主矩阵完成 {counts['case_files']} 个场景文件、{counts['total_neural_runs']} 条神经训练轨迹、"
        f"{counts['random_runs']} 条随机重启轨迹和 {counts['scip_runs']} 次 SCIP 求解，保存 {counts['primary_records']} 条主结果记录。"
        f"另有匹配采样日历的 15 秒补充对照，含 {child['counts']['case_files']} 个文件、{child['counts']['neural_runs']} 次训练、"
        f"{child['counts']['primary_records']} 条快照。合计 {combined['total_neural_runs']} 次神经训练、{combined['primary_records']} 条结果记录。"
        "固定步数与限时快照分开统计，不将同一次训练的多个截止快照算成多次训练；同一图的重复场景文件也不算新的独立图。", "",
        "## 本轮范围与控制条件", "",
        f"运行环境：Python {environment['python']}、PyTorch {environment['torch']}、CUDA {environment['cuda']}、"
        f"{environment['gpu']}，显卡驱动 {environment['driver']}。CPU 张量线程数为 {environment['cpu_threads']}，"
        f"SCIP 单线程；DHG {environment['packages']['dhg']}、NumPy {environment['packages']['numpy']}、"
        f"SciPy {environment['packages']['scipy']}、PySCIPOpt {environment['packages']['pyscipopt']}。"
        "纳入分析的训练与限时 CPU 对照串行执行，本任务主动暂停仅安排在完整运行之间；"
        "时间完整性检查未标记正式运行有明显长暂停，中断归档另述。", "",
        "收尾实验不增加新模型，也不利用已见数据重新选择主配置。弱定义仍锁定 w1024_cra、强定义仍锁定 w1024_none，惩罚系数主配置仍为 2。"
        "所有训练使用完整原始超边，结构剪枝只用于生成布尔可行域的冗余证书。", "",
        "固定步数实验覆盖两张已见的 500 节点合成图和 Cora／High／Primary 的强、弱定义，共八个场景。"
        "每场景比较 CRA 与关闭退火、三个模型种子；五个弱定义场景另比较 λ=1.1、4、8，λ=2 的 CRA 运行直接复用，合计 93 次训练。"
        "每次恰完成 1600 次参数更新，在第 1 步、每 100 步以及末步保存候选；采样集合预先固定，历史最好仅指这些候选中的第一个最大解。", "",
        "限时消融在同八个场景比较 CRA 与关闭退火，各使用一条固定 30 秒轨迹的 5／15／30 秒快照；"
        "五个弱定义场景额外分别运行 horizon=5、15 秒的压缩 CRA。合计 78 次独立训练。"
        "固定 30 秒与压缩日程只在相同截止点比较，30 秒两者本来相同，不重复计算一个新对照。", "",
        "新生成图复核覆盖 1000／2000 节点、k=3 且 m/n=2／k=5 且 m/n=4 两种结构，"
        "每格使用图种子 20261027、20261028、20261029，共 12 张新弱独立集图。每图运行三个神经种子、三个随机重启种子、单次 SCIP 和低度贪心。"
        "模型种子是同一张图上的技术重复；汇总先在图内平均模型种子，再平均不同图，不把 36 次训练当作 36 张图。"
        "三个生成种子在不同规模及结构之间重复使用，跨条件可能存在关联；12 张新图不视为相互独立同分布的样本，"
        "总体汇总及每个条件三个图的范围均为描述性结果，不计算总体显著性或置信区间。", "",
        "网络宽度为 1024，输入特征为固定的逐节点均匀随机特征，学习率 1e-4、权重衰减 5e-8、有效 dropout 为 0。"
        "相同图、定义及模型种子的全部变体核对初始网络状态与特征 SHA，并检查首个更新前的网络分数在浮点容差内一致。"
        "SHA 的配对核对不等同于从哈希重建初始参数，也不保证后续 GPU 轨迹逐位一致。"
        "主消融按完整实例、定义和模型种子归并为 24 个共享初始化组：固定步数与限时实验各覆盖同一组 24 个，"
        "共同包含 171 条变体运行，不是 48 个不同初始化组。正式匹配补充的 15 条运行连接其中 15 组；"
        "新图复核的 36 条单配置运行不计入这项变体配对数。", "",
        "固定步数 CRA 取 γ=-1+1.6×(epoch−1)/(1600−1)，首步为 −1、末步为 +0.6；限时 CRA 取 γ=-1+1.6×min(1,t/horizon)。"
        "固定步数实验控制训练量，限时实验控制名义计算预算；两者分别回答优化作用与预算内实用表现，不能合并解释为同一公平条件。", "",
        "候选均采用该训练步参数更新前、训练模式下的前向输出。固定步数最后候选是第 1600 步的更新前输出，"
        "运行本身仍完成第 1600 次参数更新。限时质量只使用截止前已完整解码的候选，实际快照发出时间和越界时长另存；"
        "最终诊断、未来解或未来求解器上界不会回填早期结果。构建、初始化哈希、训练和在线解码均计入新求解器运行时间。", "",
        f"启动源码提交：`{manifest['git_commit_at_start']}`。manifest binding：`{manifest['binding']}`。"
        "源码、完整协议、旧验证选择、环境和实例哈希均被绑定；恢复按整条轨迹保存，不拼接不同运行。", "",
        "## 退火的独立作用", "",
        "下表左侧减右侧，正数表示 CRA 产生更大的解。先按场景平均三个配对模型种子，再汇总场景；"
        "均为描述性结果，不报告显著性，也不因本轮结果改选原先锁定的配置。", "",
        "| 控制条件 | 定义 | 截止秒 | 场景数 | 平均解大小差 | 场景均值相对差 | 胜／平／负 |",
        "|---|---|---:|---:|---:|---:|---|"]
    archive = summary.get("interrupted_supplement_archive")
    if archive:
        interruption_lines = ["第一批匹配补充中，一条名义 15 秒轨迹发生长时间执行中断，完整耗时 "
            f"{archive['outliers_over_60_seconds'][0]['run_total_seconds']:.6f} 秒，具体操作系统原因尚未证实。"
            "该批全部 15 次训练、30 条快照均完整归档，整批不纳入主分析，随后按完全相同的冻结源码、协议、环境和输入重新执行全部 15 次训练。"
            "没有依据解质量挑选单条替换。主矩阵在该次中断前已退出，其耗时字段仍另行逐条复核。", "",
            f"因此，222 次训练／429 条记录指纳入分析的主矩阵与正式补充；本轮相关批次另保存 15 次／30 条中断归档，"
            f"合计保存 {combined['saved_neural_runs_including_archive']} 条神经轨迹与 {combined['saved_primary_records_including_archive']} 条主记录，"
            "预热及独立环境验证另计。"
            f"[中断证据与整批排除说明](../{archive['interruption_source']})。", ""]
        insertion = lines.index("## 本轮范围与控制条件")
        lines[insertion:insertion] = interruption_lines
    for row in summary["fixed_cra_comparisons"]["by_definition"]:
        lines.append(f"| 固定 1600 步：CRA−关闭 | {row['definition']} | — | {row['instances']} | "
            f"{fmt(row['mean_size_difference'])} | {pct(row['mean_paired_percent'])} | {'/'.join(map(str,row['wins_ties_losses']))} |")
    for row in summary["timed_cra_comparisons"]["by_definition_cutoff"]:
        lines.append(f"| 固定 30 秒轨迹：CRA−关闭 | {row['definition']} | {row['cutoff_seconds']:g} | {row['instances']} | "
            f"{fmt(row['mean_size_difference'])} | {pct(row['mean_paired_percent'])} | {'/'.join(map(str,row['wins_ties_losses']))} |")
    lines.extend(["", "固定步数的逐场景结果如下，保留全部模型种子，不以最好种子代替均值。", "",
        "| 场景 | 定义 | CRA 三种子 | 关闭退火三种子 | CRA−关闭均值 |",
        "|---|---|---|---|---:|"])
    for row in summary["fixed_cra_comparisons"]["by_case"]:
        lines.append(f"| {row['name']} | {row['definition']} | {' / '.join(map(str,row['left_sizes']))} | "
            f"{' / '.join(map(str,row['right_sizes']))} | {fmt(row['mean_size_difference'])} |")
    lines.extend(["", "## 固定时域与压缩退火", "",
        "源码审查发现，原压缩 15 秒运行的 targets=[15] 没有固定 30 秒运行中截止前 4.9 秒的额外采样及 5 秒快照；"
        "因此原 15 秒对照同时改变了时域和采样协议。原始记录全部保留作诊断，并增加独立的 targets=[5,15] 匹配补充。"
        "这项补充在自身执行前固定协议，依据为源码日历不一致，不是根据解质量挑选实例、种子或配置；父批次此前已经开始，不能称为整个项目的全局预注册。", "",
        "下面主对照为压缩日程减固定 30 秒日程：5 秒沿用原 targets=[5]，15 秒改用匹配补充 targets=[5,15] 的 15 秒快照。"
        "两者在相应截止前的名义采样及较早快照日历一致，图、模型种子、初始化、网络、解码和历史保留规则均配对。"
        "实际采样时刻、完成步数和返回延迟仍可不同；截止后才触发或完成的候选不得进入质量前缀。"
        "这项对照考察时域设置，与单纯开启／关闭 CRA 的问题分开。", "",
        "| 截止秒 | 场景数 | 平均大小差 | 场景均值相对差 | 胜／平／负 |",
        "|---:|---:|---:|---:|---|"])
    for row in summary["primary_schedule_comparisons"]["by_cutoff"]:
        lines.append(f"| {row['cutoff_seconds']:g} | {row['instances']} | {fmt(row['mean_size_difference'])} | "
                     f"{pct(row['mean_paired_percent'])} | {'/'.join(map(str,row['wins_ties_losses']))} |")
    lines.extend(["", "| 场景 | 截止秒 | 压缩 CRA 三种子 | 固定 30 秒 CRA 三种子 | 差值均值 |",
        "|---|---:|---|---|---:|"])
    for row in summary["primary_schedule_comparisons"]["by_case"]:
        lines.append(f"| {row['name']} | {row['cutoff_seconds']:g} | {' / '.join(map(str,row['left_sizes']))} | "
                     f"{' / '.join(map(str,row['right_sizes']))} | {fmt(row['mean_size_difference'])} |")
    lines.extend(["", "原未匹配 15 秒运行保留为以下诊断，它不能单独归因为时域变化。", "",
        "| 诊断 | 场景数 | 平均大小差 | 场景均值相对差 | 胜／平／负 |",
        "|---|---:|---:|---:|---|"])
    for row in summary["unmatched15_diagnostics"]["by_cutoff"]:
        lines.append(f"| 未匹配 targets=[15]−固定30，截止15秒 | {row['instances']} | {fmt(row['mean_size_difference'])} | "
                     f"{pct(row['mean_paired_percent'])} | {'/'.join(map(str,row['wins_ties_losses']))} |")
    for row in child["comparisons"]["by_cutoff"]:
        if row["cutoff_seconds"] == 5:
            lines.append(f"| 匹配 horizon15−固定30，早期截止5秒 | {row['instances']} | {fmt(row['mean_size_difference'])} | "
                         f"{pct(row['mean_paired_percent'])} | {'/'.join(map(str,row['wins_ties_losses']))} |")
    lines.extend(["", "匹配补充的全部两个截止快照及原未匹配数据都在机器可读汇总中保留，未按结果删种子。"
        f"补充 manifest binding：`{child['manifest_binding']}`；父 binding 与主批次一致。"
        "见 [补充协议](../experiments/schedule-match-20261020.json)。", "",
        "## 惩罚系数敏感性与保证边界", "",
        "以下各值均大于 1，因此共享无权二元目标全局最优可行性证明；该证明不保证连续网络优化、阈值输出或限步优化一定可行或最优。"
        "敏感性只比较固定 1600 步、同一 CRA 日程与配对初始化；λ=2 始终为预声明主配置，其他值不用于事后选参。", "",
        "| λ | 场景数 | 相对 λ=2 平均解大小差 | 场景均值相对差 | 胜／平／负 |",
        "|---:|---:|---:|---:|---|"])
    for row in summary["penalty_comparisons"]["by_penalty"]:
        lines.append(f"| {row['penalty']:g} | {row['instances']} | {fmt(row['mean_size_difference'])} | "
                     f"{pct(row['mean_paired_percent'])} | {'/'.join(map(str,row['wins_ties_losses']))} |")
    lines.extend(["", "| 场景 | λ | 三种子解大小 | 原始可行率 | 平均移除／补入 | 最好−最新 |",
        "|---|---:|---|---:|---|---:|"])
    for row in summary["fixed_rows"]:
        if row["definition"] == "weak" and row["annealing"] == "cra":
            lines.append(f"| {row['name']} | {row['penalty']:g} | {' / '.join(map(str,row['sizes']))} | "
                f"{ratio(row['raw_feasible_rate'])} | {fmt(row['removed_mean'])}／{fmt(row['added_mean'])} | {fmt(row['mean_best_minus_latest'])} |")
    lines.extend(["", "两节点、单条二节点边的全部四种二元赋值被精确枚举。λ=0.5 时存在不可行的唯一最小解；"
        "λ=1 时不可行赋值与可行赋值并列最小，不能保证每个全局最小解可行；λ=1.1／2／4／8 的所有二元最小解均可行。"
        "这些反例只用于界定证明，不放开生产求解接口的 λ>1 限制。", "",
        "## 十二张新生成图的规模复核", "",
        "30 秒时，按图内三个神经种子均值再跨 12 张图汇总，神经方法相对低度贪心、随机重启和 SCIP 截止可行解的平均差异分别为 " +
        "、".join(pct(overall["comparisons"][method]["mean_paired_percent"]) for method in ("degree", "random_restart", "scip")) +
        "。这组新图用于复核锁定配置，不代表任意超图分布的普遍优势。", "",
        "| 节点数 | k | m/n | 图数 | 对贪心平均差异 | 对随机重启平均差异 | 对 SCIP 平均差异 | 对 SCIP 胜／平／负 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in summary["replication_groups"]:
        lines.append(f"| {row['n']} | {row['k']} | {row['density_m_over_n']} | {row['graphs']} | "
            f"{pct(row['comparisons']['degree']['mean_paired_percent'])} | {pct(row['comparisons']['random_restart']['mean_paired_percent'])} | "
            f"{pct(row['comparisons']['scip']['mean_paired_percent'])} | {'/'.join(map(str,row['comparisons']['scip']['wins_ties_losses']))} |")
    lines.extend(["", "| 实例 | 贪心 | 随机重启三种子 | 神经三种子 | SCIP 截止解／上界 | 截止前证明最优 |",
        "|---|---:|---|---|---|---|"])
    for row in summary["replication_rows"]:
        lines.append(f"| {row['name']} | {row['degree_size']} | {' / '.join(map(str,row['random_sizes']))} | "
            f"{' / '.join(map(str,row['neural_sizes']))} | {row['scip_size']}／{fmt(row['scip_upper_bound'],3)} | "
            f"{'是' if row['scip_optimality_proven'] else '否'} |")
    lines.extend(["", "SCIP 的实际完成时间、晚到最终解、终端状态与截止前解分开保存。"
        "只有截止前完整得到最优证明时才标记已证明最优；没有证明的较大可行解不称为最大独立集。"
        "图中的范围表示同一规模及结构下三个生成种子的图均值最小值到最大值，不是置信区间。", "",
        "## 实际耗时、发出延迟与显存记录", "",
        "固定步数耗时仅作描述。下面限时结果的质量仍按名义截止前缀计算，实际返回延迟单列；"
        "发出时已完成更新数可以包含截止后的最后一步，不能当作严格截止前更新次数。", "",
        "| 范围 | 退火 | 时域秒 | 截止秒 | 快照数 | 发出时间中位秒 | 最小—最大秒 | 最大晚发秒 |",
        "|---|---|---:|---:|---:|---:|---|---:|"])
    timed_neural = [row for row in summary["neural_rows"] if row["group"] != "fixed"] + child["neural_rows"]
    for key in sorted({(row["group"], row["annealing"], row["horizon_seconds"], row["cutoff_seconds"]) for row in timed_neural}):
        values = [row for row in timed_neural if (row["group"], row["annealing"], row["horizon_seconds"], row["cutoff_seconds"]) == key]
        seconds = [row["total_seconds"] for row in values]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]:g} | {key[3]:g} | {len(values)} | {st.median(seconds):.4f} | "
                     f"{min(seconds):.4f}—{max(seconds):.4f} | {max(row['overshoot_seconds'] for row in values):.4f} |")
    lines.extend(["", "| 训练组 | 独立训练数 | 完整运行中位秒 | 最小—最大秒 | 完成步数中位 | 峰值分配显存中位／最大 MiB |",
        "|---|---:|---:|---|---:|---|"])
    for group, profile in {**summary["neural_run_profile"], "schedule_match": child["neural_run_profile"]}.items():
        seconds, epochs, peak = profile["run_total_seconds"], profile["epochs_completed"], profile["peak_allocated_bytes"]
        lines.append(f"| {group} | {seconds['count']} | {seconds['median']:.4f} | {seconds['min']:.4f}—{seconds['max']:.4f} | "
                     f"{fmt(epochs['median'],1)} | {peak['median']/2**20:.2f}／{peak['max']/2**20:.2f} |")
    scip_seconds = [row["scip_actual_seconds"] for row in summary["replication_rows"]]
    random_seconds = [value for row in summary["replication_rows"] for value in row["random_actual_seconds"]]
    lines.extend(["", f"新图 SCIP 完整运行时间为 {min(scip_seconds):.4f}—{max(scip_seconds):.4f} 秒，"
        f"中位数 {st.median(scip_seconds):.4f} 秒；随机重启实际快照发出时间为 {min(random_seconds):.4f}—{max(random_seconds):.4f} 秒。"
        "显存记录为 PyTorch 的峰值已分配字节，不等同于整个进程或显卡全部占用；"
        "各组混有不同实例和设置，该表不用于单独归因某个实现优化。", "",
        "全量检查主矩阵与正式补充的完整运行、快照发出和保存候选／试次／求解器事件时间轴。"
        "诊断长暂停时采用明确的粗粒度门槛：限时总耗时或快照发出超过 max(60 秒, 两倍目标)、固定步数总耗时超过 600 秒、"
        "保存事件空档或未归入组件的耗时超过 60 秒。发现异常会拒绝完成声明，不自动删掉不利结果。", "",
        "| 范围 | 已核验完整运行数 | 完整耗时最小—最大秒 | 最大保存事件空档秒 | 最大快照晚发秒 | 长暂停标记数 |",
        "|---|---:|---|---:|---:|---:|"])
    for label, timing in (("主矩阵", summary["timing_integrity"]), ("正式匹配补充", child["timing_integrity"])):
        lines.append(f"| {label} | {timing['run_count']} | {timing['minimum_run_total_seconds']:.4f}—{timing['maximum_run_total_seconds']:.4f} | "
                     f"{timing['largest_saved_gap']['max_evidence_gap_seconds']:.4f} | "
                     f"{timing['largest_lateness']['lateness_seconds']:.4f} | {len(timing['gross_interruption_flags'])} |")
    lines.extend(["", "上述纳入批次未发现与归档相似的长时间中断；归档的唯一异常轨迹及所有原始快照另行保留，不混入这张表。" if archive else
        "上述纳入批次的保存时间字段通过长暂停诊断。", "",
        "## 历史保留、后处理和原始可行性", "",
        "保留历史最好与使用截止前最新候选可直接由同一轨迹比较，不另造一次训练。"
        "下面列出 λ=2 的固定 1600 步 CRA／关闭退火，以及固定 30 秒轨迹的末个截止；原始可行率均针对被选的历史候选。"
        "历史最好与最新之差衡量保留规则，最好与初始之差同时包含优化和多时刻选择，不能单独等同于终末模型的学习增益。", "",
        "| 控制 | 场景 | 定义 | 退火 | 历史最好原始→修复→扩充均值 | 最好原始可行率 | 最新原始可行率 | 最好−初始 | 最好−最新 |",
        "|---|---|---|---|---|---:|---:|---:|---:|"])
    selected = [("1600步", row) for row in summary["fixed_rows"] if row["penalty"] == 2]
    selected.extend(("30秒", row) for row in summary["timed_rows"] if row["horizon_seconds"] == row["cutoff_seconds"] == 30)
    for control, row in selected:
        lines.append(f"| {control} | {row['name']} | {row['definition']} | {row['annealing']} | "
            f"{fmt(row['raw_size_mean'])}→{fmt(row['repaired_size_mean'])}→{fmt(row['mean_size'])} | "
            f"{ratio(row['raw_feasible_rate'])} | {ratio(row['latest_raw_feasible_rate'])} | "
            f"{fmt(row['mean_best_minus_initial'])} | {fmt(row['mean_best_minus_latest'])} |")
    lines.extend(["", "空集兜底与模型输出分开统计。历史最好按修复扩充后的解大小选择，可能选中一个较早、原始阈值不可行的候选；"
        "因此历史最好原始可行率不能当作末次网络原始可行率。修复与扩充后保证可行且极大，极大不等于最大。", "",
        "## 真实图的弱约束结构证书", "",
        "独立审计通过真子集枚举重建包含关系最小反链，未调用生产剪枝器。"
        "每条删除边均保存一个保留真子边见证：若真子边已不能全选，则包含它的超边也不能全选；据此保持整个弱 MIS 二元可行集合。", "",
        "| 数据 | 原始超边 | 保留超边 | 删除超边 | 保留边阶数分布 |",
        "|---|---:|---:|---:|---|"])
    for row in summary["structural"]["weak_redundancy"]:
        histogram = ", ".join(f"k={k}: {value}" for k, value in row["retained_order_histogram"].items())
        lines.append(f"| {row['dataset']} | {row['original_edges']} | {row['retained_edges']} | {row['removed_edges']} | {histogram} |")
    lines.extend(["", "High 的全部高阶弱约束由二节点边蕴含；Primary 除二节点边外还保留五条三节点约束；"
        "Cora 在更一般的真子集检查下也有部分冗余。所有训练仍使用原图。"
        "删除这些约束会改变连续多项式、梯度与消息传播，不构成连续能量等价，也不适用于直接删掉强 MIS 的高阶超边。"
        "真实文件仍按 BIPNN 公开仓库版本解释，Cora 与论文表格的数量差异及原采集身份限制见 [数据来源审计](data-provenance.md)。", "",
        "## 独立复核与复现范围", "",
        f"主矩阵与匹配补充联合重算 {combined['neural_candidates_audited']} 个带网络分数的候选，逐个验证阈值、删除顺序、修复集合、扩充集合、"
        "可行性、极大性与独立多项式损失；检查精确日程、首次并列选优、固定步数采样、时间前缀、初始化配对及主记录身份。"
        "随机对照重放 RNG 到首次及每次严格改进试次，检查全部试次日志的前缀最大值；SCIP 的过去解、过去上界和晚到诊断分别审计，"
        "全部求解器界均不得低于任一已保存可行候选。", "",
        "全新隔离环境的安装与端到端验证见 [干净环境复现报告](clean-reproduction-20261020.md)，"
        "具体日志以该报告列明的最终运行版本为准。验证范围为新环境安装、61 项测试且无跳过、弱／强各三个种子 200 步的 GPU 小图、"
        "SCIP 精确对照及 14 条保存小图记录的报告核验；没有在新环境重跑全部实验矩阵。安装脚本补入此前遗漏的 PySCIPOpt 依赖，新的环境结果不覆盖历史批次。"
        "DHG 的 torch<2 包元数据冲突仍显式保留，只验证实际使用的算子路径，不据此声称整个 DHG 与当前 PyTorch 全部兼容。", "",
        "```powershell", ".venv/Scripts/python.exe scripts/run_completion.py --output results/completion-rerun",
        ".venv/Scripts/python.exe scripts/run_schedule_match.py --parent results/completion-rerun --output results/schedule-match-rerun",
        ".venv/Scripts/python.exe scripts/audit_completion.py results/completion-rerun --schedule-match results/schedule-match-rerun --report docs/completion-rerun.md", "```", "",
        "这些命令按冻结计划运行完整收尾批次。恢复会拒绝源码、协议、输入或环境变化。"
        "补充计划还绑定父 manifest；若父 binding 改变，应另行固定新的补充协议，不能解除这个校验。"
        "固定种子不承诺其他设备上逐位相同结果；尤其限时实验会因设备与负载变化完成不同步数。", "",
        "本轮把原定必要实验补齐，但毕业论文仍需将方法、实验分析、引用、学校格式及导师意见落实到正文。"
        "本报告不将已见数据的消融当作新的未见测试，不把技术重复当作独立图样本，也不将局部预算优势解释为普遍最优性或统计显著性。", ""])
    return "\n".join(lines)


def build_summary(directory, manifest, *, initial_registry=None):
    expected = expected_inputs(directory, manifest["plan"])
    equal(sorted(manifest["inputs"]), sorted(expected), "Bound case set differs from complete plan")
    actual = {str(path.relative_to(directory / "cases").with_suffix("")).replace("\\", "/")
              for path in (directory / "cases").glob("*/*.json")}
    require(actual == set(expected), f"Incomplete/extra cases: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}")
    registry = {} if initial_registry is None else initial_registry
    reports, rows = [], []
    for case_id, (group, name, mode, instance) in expected.items():
        require(manifest["inputs"][case_id] == fingerprint(instance), "Manifest and independently checked input differ")
        report = audit_case(directory / "cases" / group / f"{name}_{mode}.json", manifest, initial_registry=registry)
        require(report["group"] == group and report["definition"] == mode,
                "Case identity differs from the predeclared input definition/group")
        equal(report["instance"], instance, "Case differs from separately saved input")
        reports.append(report)
        rows.extend(neural_rows(report))
        print(f"AUDITED {case_id}: {len(report['runs'])} whole runs, {len(report['records'])} primary records", flush=True)
    fixed, timed = aggregate_neural(rows, "fixed"), aggregate_neural(rows, "timed")
    replication = replication_rows(reports, rows)
    structure = audit_structural(directory, manifest)
    runs = [run for report in reports for run in report["runs"]]
    neural = [run for run in runs if run["method"] in {"bipnn_fixed_epochs_completion", TIMED_METHOD}]
    randoms = [run for run in runs if run["method"] == "random_restart_same_decoder"]
    scips = [run for run in runs if run["method"] == "scip_milp_greedy_start"]
    counts = {"fixed_cases": sum(report["group"] == "fixed" for report in reports),
        "fixed_neural_runs": sum(report["group"] == "fixed" and run["method"] == "bipnn_fixed_epochs_completion"
                                  for report in reports for run in report["runs"]),
        "timed_cases": sum(report["group"] == "timed" for report in reports),
        "timed_neural_runs": sum(len(report["runs"]) for report in reports if report["group"] == "timed"),
        "replication_cases": len(replication),
        "replication_neural_runs": sum(report["group"] == "replication" and run["method"] == TIMED_METHOD
                                        for report in reports for run in report["runs"]),
        "random_runs": len(randoms), "scip_runs": len(scips), "total_neural_runs": len(neural),
        "case_files": len(reports), "primary_records": sum(len(report["records"]) for report in reports)}
    equal(counts, manifest["plan"]["expected_counts"], "Actual suite counts differ from predeclared complete suite")
    require(len(rows) == 303 and len(fixed) == 31 and len(timed) == 58 and len(replication) == 12,
            "Derived neural summary dimensions differ")
    counts.update({"neural_primary_rows": len(rows), "fixed_summary_rows": len(fixed), "timed_summary_rows": len(timed),
        "neural_candidates_audited": sum(sum("scores" in c for c in run["candidates"]) for run in neural),
        "paired_initialization_groups": sum(entry["runs"] > 1 for entry in registry.values()),
        "paired_neural_runs": sum(entry["runs"] for entry in registry.values() if entry["runs"] > 1),
        "random_trials_logged": sum(len(run["trial_log"]) for run in randoms),
        "random_improving_candidates_audited": sum(len(run["candidates"]) - 1 for run in randoms),
        "scip_incumbent_events_audited": sum(len(run["incumbent_events"]) for run in scips),
        "scip_bound_events_audited": sum(len(run["bound_events"]) for run in scips)})
    fixed_initializations = {(row["name"], row["definition"], row["seed"]) for row in rows if row["group"] == "fixed"}
    timed_initializations = {(row["name"], row["definition"], row["seed"]) for row in rows if row["group"] == "timed"}
    counts.update({"fixed_initialization_groups": len(fixed_initializations),
                   "timed_initialization_groups": len(timed_initializations),
                   "cross_group_shared_initializations": len(fixed_initializations & timed_initializations)})
    require(counts["paired_initialization_groups"] == 24 and counts["paired_neural_runs"] == 171,
            "Paired ablation initialization coverage differs")
    require(fixed_initializations == timed_initializations and len(fixed_initializations) == 24,
            "Fixed and timed ablations do not cover the same 24 initialization groups")
    return {"complete": False, "main_suite_complete": True, "manifest_binding": manifest["binding"],
        "audit_source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("scripts/audit_completion.py", "scripts/summarize_controlled.py", "scripts/summarize_anytime.py")},
        "counts": counts,
        "neural_rows": rows, "fixed_rows": fixed, "timed_rows": timed,
        "timing_integrity": audit_timing_ranges(reports),
        **comparisons(fixed, timed), "replication_rows": replication,
        "replication_groups": replication_groups(replication), "replication_overall": replication_group(replication),
        "structural": structure,
        "neural_run_profile": {group: {field: distribution([run[field] for report in reports if report["group"] == group
            for run in report["runs"] if run["method"] in {"bipnn_fixed_epochs_completion", TIMED_METHOD}])
            for field in ("run_total_seconds", "build_seconds", "training_seconds", "online_decode_seconds", "epochs_completed", "peak_allocated_bytes")}
            for group in ("fixed", "timed", "replication")},
        "limits": ["Descriptive bounded suite; no significance claim or replacement hyperparameter selection",
                   "Fixed1600 controls updates; timed controls wall-clock cutoff; do not merge causal interpretations",
                   "First maximum among predeclared sampled candidates, not all possible network outputs",
                   "Real and ablation graphs seen before; twelve replication graphs use three new graph seeds reused across conditions",
                   "Model seeds are technical repeats; average within graph before across graph",
                   "Shared graph seeds across size/structure conditions may induce dependence; no IID, significance or confidence-interval claim",
                   "Boolean weak pruning certificates do not preserve the continuous loss/message representation",
                   "Late candidates, terminal status and future dual bounds excluded from cutoff quality"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/completion-20261020.md")
    parser.add_argument("--schedule-match", type=Path, help="Required for a final global summary; separately bound matched-calendar supplement")
    parser.add_argument("--interrupted-archive", type=Path, help="Excluded complete first supplement; defaults to a sibling suffixed -interrupted if present")
    parser.add_argument("--partial-case", type=Path, help="Read one saved case only; never writes a final summary")
    args = parser.parse_args()
    manifest = manifest_checked(args.directory)
    if args.partial_case:
        report = audit_case(args.partial_case, manifest, complete=False)
        print(json.dumps({"complete": False, "partial_case": report["case_id"], "runs_audited": len(report["runs"]),
                          "primary_records_audited": len(report["records"])}, ensure_ascii=False))
        return
    require(args.schedule_match is not None and (args.schedule_match / "manifest.json").exists(),
            "Final global audit requires --schedule-match with its completed supplementary manifest")
    initial_registry = {}
    summary = build_summary(args.directory, manifest, initial_registry=initial_registry)
    child = audit_schedule_match(args.schedule_match, args.directory, manifest, summary, initial_registry)
    archive_path = args.interrupted_archive or args.schedule_match.with_name(args.schedule_match.name + "-interrupted")
    if args.directory.name == "completion-20261020":
        require((archive_path / "interruption.json").exists(), "The recorded interrupted attempt must be retained and disclosed for this batch")
    archive = audit_interrupted_archive(archive_path, args.schedule_match, manifest, initial_registry) if archive_path.exists() else None
    attach_schedule_match(summary, child)
    summary["interrupted_supplement_archive"] = archive
    excluded_runs = archive["counts"]["neural_runs"] if archive else 0
    excluded_records = archive["counts"]["primary_records"] if archive else 0
    summary["combined_counts"].update({"excluded_archived_neural_runs": excluded_runs,
        "excluded_archived_primary_records": excluded_records,
        "saved_neural_runs_including_archive": summary["combined_counts"]["total_neural_runs"] + excluded_runs,
        "saved_primary_records_including_archive": summary["combined_counts"]["primary_records"] + excluded_records})
    require(summary["complete"] is True, "Main and supplementary audits did not both pass")
    document = report_text(summary, manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(document, encoding="utf8")
    temporary.replace(args.report)
    write_json_atomic(args.directory / "summary.json", summary)
    print(json.dumps({"complete": True, **summary["counts"], "combined_counts": summary["combined_counts"],
                      "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, StopIteration) as error:
        raise SystemExit(f"Audit refused; no new complete summary was produced: {error}")
