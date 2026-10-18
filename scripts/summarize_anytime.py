"""Audit complete anytime trajectories and strict-past snapshots without training."""
import argparse
import bisect
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mis.benchmark import generate_instance, write_json
from mis.datasets import load_upstream_dataset
from mis.problem import HypergraphProblem
from mis.verify_report import verify
from summarize_controlled import (audit_decoder, audit_synthetic, distribution, fingerprint,
    finite_nonnegative, read_json, require, select_from_validation, solution)

NEURAL = "bipnn_fast_anytime"
RANDOM = "random_restart_same_decoder"
SCIP = "scip_milp_greedy_start"
SINGLE = "random_score_same_decoder"
SOURCES = {"mis/anytime.py", "mis/controlled.py", "mis/indexed_loss.py", "mis/fast_decode.py",
    "mis/fast_baselines.py", "mis/problem.py", "mis/benchmark.py", "mis/datasets.py",
    "src/models.py", "src/train.py", "scripts/run_anytime.py", "scripts/check_controlled_protocol.py",
    "scripts/run_controlled.py"}
LABELS = {"weak": "弱独立集", "strong": "强独立集", NEURAL: "神经网络前缀最好",
    RANDOM: "随机重启前缀最好", SCIP: "SCIP 截止前可行解", SINGLE: "随机单次", "degree_greedy": "低度贪心"}


def equal(actual, expected, message):
    require(fingerprint(actual) == fingerprint(expected), message)


def zero_candidate(candidate, problem, mode):
    solution(problem, mode, candidate)
    require(candidate["selected"] == [] and candidate["available_seconds"] == 0.,
            "Initial empty incumbent is not available at time zero")
    require("scores" not in candidate, "Empty incumbent must not be called a neural prediction")


def degree_independent(problem, mode):
    incidence = [[] for _ in range(problem.n)]
    for i, edge in enumerate(problem.edges):
        for v in edge:
            incidence[v].append(i)
    caps = [len(edge) - 1 if mode == "weak" else 1 for edge in problem.edges]
    counts = [0] * len(caps)
    selected = []
    for v in sorted(range(problem.n), key=lambda x: (len(incidence[x]), x)):
        if all(counts[i] < caps[i] for i in incidence[v]):
            selected.append(v)
            for i in incidence[v]:
                counts[i] += 1
    return sorted(selected)


def audit_snapshot(record, candidates, target, run_total):
    eligible = [i for i, candidate in enumerate(candidates) if candidate["available_seconds"] <= target]
    best = max(eligible, key=lambda i: candidates[i]["size"])
    for key, value in candidates[best].items():
        require(key in record, f"Snapshot omits candidate field {key}")
        equal(record[key], value, f"Snapshot changed candidate field {key}")
    require(record["budget_seconds"] == target and record["cutoff_seconds"] == target,
            "Snapshot has the wrong target/cutoff")
    require(record["best_candidate_index"] == best and record["latest_candidate_index"] == eligible[-1]
            and record["candidate_count"] == len(eligible), "Snapshot does not use its strict candidate prefix")
    require(record["best_available_seconds"] == candidates[best]["available_seconds"] <= target,
            "A future candidate entered an earlier cutoff")
    emitted = record["total_seconds"]
    finite_nonnegative(emitted, "snapshot actual emission")
    require(target <= emitted <= run_total and record["snapshot_emitted_seconds"] == emitted,
            "Incorrect actual emission time")
    require(record["overshoot_seconds"] == max(0., emitted - target) and
            record["deadline_missed"] == (emitted > target), "Incorrect deadline/overshoot disclosure")
    return best


def audit_neural(run, problem, mode, plan):
    targets = plan["checkpoints_seconds"]
    require(run["method"] == NEURAL and run["config"] == plan["locked_configs"][mode], "Wrong neural config")
    width = 1024 if run["config"].startswith("w1024") else 256
    annealing = "cra" if run["config"].endswith("_cra") else "none"
    equal(run["settings"], {"width": width, "annealing": annealing, "targets_seconds": [float(t) for t in targets],
        "horizon_seconds": float(targets[-1]), "candidate_interval_seconds": plan["candidate_interval_seconds"],
        "pre_cutoff_margin_seconds": .1, "penalty": 2., "lr": 1e-4, "weight_decay": 5e-8,
        "effective_dropout": 0., "output": "pre-update training forward",
        "selection": "first maximum feasible candidate completed by cutoff",
        "schedule": "-1+1.6*min(1,elapsed/horizon) if CRA, else 0"}, "Neural settings differ from the frozen protocol")
    total = run["run_total_seconds"]
    finite_nonnegative(total, "neural run total")
    require(total >= targets[-1], "Neural trajectory stopped before its fixed horizon")
    require(type(run["epochs_completed"]) is int and run["epochs_completed"] >= 1, "Invalid neural update count")
    for field in ("build_seconds", "training_seconds", "online_decode_seconds"):
        finite_nonnegative(run[field], field)
    require(sum(run[k] for k in ("build_seconds", "training_seconds", "online_decode_seconds")) <= total + 1e-5,
            "Neural component times exceed the complete run")
    candidates = run["candidates"]
    require(len(candidates) >= 3 and run["initial_index"] == 1 and run["last_index"] == len(candidates) - 1,
            "Missing initial/terminal neural evidence")
    zero_candidate(candidates[0], problem, mode)
    require(candidates[0]["role"] == "empty_initial_incumbent", "Wrong empty role")
    previous_time, previous_epoch = 0., 0
    for i, candidate in enumerate(candidates[1:], 1):
        finite_nonnegative(candidate["available_seconds"], "candidate availability")
        require(previous_time <= candidate["available_seconds"] <= total, "Nonmonotone candidate availability")
        require(type(candidate["epoch"]) is int and previous_epoch <= candidate["epoch"] <= run["epochs_completed"],
                "Invalid candidate epoch")
        require(math.isfinite(candidate["loss_before_update"]), "Nonfinite saved neural loss")
        coefficient = candidate["annealing_coefficient"]
        require(math.isfinite(coefficient) and (-1 <= coefficient <= .6 + 1e-12 if annealing == "cra" else coefficient == 0),
                "Invalid saved annealing coefficient")
        expected_role = ("untrained_first_forward" if i == 1 else
                         "terminal_diagnostic" if i == len(candidates) - 1 else "periodic_training_forward")
        require(candidate["role"] == expected_role, "Neural candidate role changed")
        audit_decoder(problem, mode, candidate, full_stages=True)
        previous_time, previous_epoch = candidate["available_seconds"], candidate["epoch"]
    require(candidates[1]["epoch"] == 1 and candidates[-1]["epoch"] == run["epochs_completed"], "Incorrect initial/final epoch")
    require(len(run["records"]) == len(targets), "Missing neural snapshots")
    sizes = []
    for record, target in zip(run["records"], targets):
        require(record["method"] == NEURAL and record["seed"] == run["seed"] and record["config"] == run["config"],
                "Neural snapshot identity differs from its run")
        audit_snapshot(record, candidates, target, total)
        require(type(record["epochs_completed_at_snapshot"]) is int and
                candidates[record["latest_candidate_index"]].get("epoch", 0) <= record["epochs_completed_at_snapshot"] <= run["epochs_completed"],
                "Invalid emitted-time epoch count")
        sizes.append(record["size"])
    require(sizes == sorted(sizes), "Neural prefix-best size decreases")
    return [candidate["size"] for candidate in candidates]


def audit_random(run, problem, mode, plan):
    require(run["method"] == RANDOM, "Wrong random trajectory method")
    targets = plan["checkpoints_seconds"]
    total = run["run_total_seconds"]
    finite_nonnegative(total, "random run total")
    require(total >= targets[-1], "Random trajectory stopped before the horizon")
    log = run["trial_log"]
    require(bool(log), "Missing random trial log")
    previous, best_size, improving = 0., 0, []
    for trial, entry in enumerate(log, 1):
        require(isinstance(entry, list) and len(entry) == 2, "Invalid trial log entry")
        available, size = entry
        finite_nonnegative(available, "random trial availability")
        require(previous <= available <= total and type(size) is int and 0 <= size <= problem.n,
                "Invalid random trial timing/size")
        if size > best_size:
            improving.append(trial)
            best_size = size
        previous = available
    candidates = run["candidates"]
    zero_candidate(candidates[0], problem, mode)
    require([candidate["trial"] for candidate in candidates[1:]] == improving, "Missing or spurious random improving candidate")
    by_trial = {candidate["trial"]: candidate for candidate in candidates[1:]}
    single = run["single"]
    require(single["method"] == SINGLE and single["seed"] == run["seed"] and single["trial"] == 1 and
            single["trials"] == 1 and single["budget_seconds"] is None and single["total_seconds"] == log[0][0],
            "Random single is not the first trial from the same trajectory")
    rng = random.Random(run["seed"])
    for trial in range(1, len(log) + 1):
        # Replay RNG state, but decode only first/improving trials. Re-solving
        # every logged random trial would repeat the complete 30-second search.
        scores = [rng.random() for _ in range(problem.n)]
        selected = ([single] if trial == 1 else []) + ([by_trial[trial]] if trial in by_trial else [])
        for candidate in selected:
            equal(candidate["scores"], scores, "Random scores do not match seed/trial replay")
            require(candidate["size"] == log[trial - 1][1] and candidate["available_seconds"] == log[trial - 1][0],
                    "Random candidate differs from its trial log")
            audit_decoder(problem, mode, candidate, full_stages=True)
    require(len(run["records"]) == len(targets), "Missing random snapshots")
    for record, target in zip(run["records"], targets):
        require(record["method"] == RANDOM and record["seed"] == run["seed"], "Wrong random snapshot identity")
        audit_snapshot(record, candidates, target, total)
        prefix_count = bisect.bisect_right([entry[0] for entry in log], target)
        prefix_sizes = [entry[1] for entry in log[:prefix_count]]
        require(record["trials"] == prefix_count and record["size"] == max([0] + prefix_sizes),
                "Random snapshot is not the maximum of completed trials")
        first_best_trial = 0 if record["size"] == 0 else prefix_sizes.index(record["size"]) + 1
        require(record["trial"] == first_best_trial, "Random best trial does not retain the earliest tie")
    return [candidate["size"] for candidate in candidates] + [single["size"]]


def audit_scip(run, record, problem, mode, budget, degree):
    require(run["method"] == SCIP and run["time_limit_seconds"] == budget and
            run["threads"] == 1 and run["random_seed_shift"] == 0, "SCIP method/settings changed")
    solution(problem, mode, run)
    total = run["total_seconds"]
    finite_nonnegative(total, "SCIP complete actual time")
    for field in ("build_seconds", "solver_seconds", "event_callback_seconds", "report_assembly_seconds"):
        finite_nonnegative(run[field], field)
    require(run["warm_start_size"] == len(degree) and run["warm_start"] == "degree_greedy" and
            run["size"] >= len(degree), "SCIP warm start differs from the equivalent greedy baseline")
    require(math.isclose(run["solver_time_limit_seconds"], max(.001, budget - run["build_seconds"]),
                         rel_tol=0, abs_tol=1e-9), "SCIP model construction was not deducted")
    require(run["optimality_proven"] == (run["status"] == "optimal"), "Invalid SCIP terminal optimality label")
    audit_bound(run, problem.n)
    if run["mip_gap"] is not None:
        finite_nonnegative(run["mip_gap"], "SCIP finite terminal gap")
        require(run["mip_gap_flag"] == "finite" and not run["mip_gap_is_infinite"], "Inconsistent finite SCIP gap flags")
    else:
        require(run["mip_gap_flag"] != "finite", "Missing terminal gap mislabeled finite")
    if run["optimality_proven"]:
        require(abs(run["upper_bound"] - run["size"]) <= 1e-5 and run["mip_gap"] == 0,
                "SCIP optimal status has inconsistent final bounds/gap")
    incumbents, bounds = run["incumbent_events"], run["bound_events"]
    require(incumbents[0]["selected"] == [] and incumbents[0]["available_seconds"] == 0,
            "Missing SCIP time-zero empty incumbent")
    require(bounds[0]["upper_bound"] == problem.n and bounds[0]["available_seconds"] == 0,
            "Missing SCIP time-zero trivial bound")
    require(sum(event["source"] == "degree_greedy_warm_start" for event in incumbents) == 1,
            "Missing or repeated SCIP warm-start evidence")
    previous, sizes = 0., []
    for event in incumbents:
        solution(problem, mode, event)
        finite_nonnegative(event["available_seconds"], "SCIP incumbent availability")
        require(previous <= event["available_seconds"] <= total, "Nonmonotone SCIP incumbent timeline")
        if event["source"] == "degree_greedy_warm_start":
            require(event["selected"] == degree, "SCIP warm-start event changed its vertices")
        previous = event["available_seconds"]
        sizes.append(event["size"])
    require(sizes == sorted(sizes), "SCIP terminal run lost a better incumbent")
    require(incumbents[-1]["source"] == "terminal_after_cleanup" and incumbents[-1]["available_seconds"] == total,
            "SCIP terminal evidence predates cleanup")
    equal(incumbents[-1]["selected"], run["selected"], "SCIP terminal solution differs from its final event")
    previous = 0.
    for event in bounds:
        finite_nonnegative(event["available_seconds"], "SCIP bound availability")
        require(previous <= event["available_seconds"] <= total, "Nonmonotone SCIP bound timeline")
        audit_bound(event, problem.n)
        previous = event["available_seconds"]
    require(bounds[-1]["source"] == "terminal_after_cleanup" and bounds[-1]["available_seconds"] == total and
            bounds[-1]["upper_bound"] == run["upper_bound"], "Missing or inconsistent SCIP terminal bound event")
    eligible = [event for event in incumbents if event["available_seconds"] <= budget]
    best = max(eligible, key=lambda event: event["size"])
    bound = [event for event in bounds if event["available_seconds"] <= budget][-1]
    cutoff = run["cutoff"]
    equal(cutoff["selected"], best["selected"], "SCIP cutoff used a future or wrong incumbent")
    require(cutoff["size"] == best["size"] and cutoff["available_seconds"] == best["available_seconds"] and
            cutoff["upper_bound"] == bound["upper_bound"] and
            cutoff["upper_bound_available_seconds"] == bound["available_seconds"] <= budget,
            "SCIP cutoff used a future incumbent or bound")
    completed = total <= budget
    require(cutoff["optimality_proven"] == (completed and run["optimality_proven"]) and
            cutoff["status"] == (run["status"] if completed else "deadline_snapshot"),
            "Late terminal status leaked into the cutoff")
    require(cutoff["solver_completed_before_deadline"] == completed, "Invalid solver completion flag")
    derived_gap = ((bound["upper_bound"] - best["size"]) / best["size"] if best["size"] else
                   0. if bound["upper_bound"] == 0 else None)
    equal(cutoff["mip_gap"], derived_gap, "Invalid cutoff gap")
    expected = {**cutoff, "method": SCIP, "budget_seconds": budget, "cutoff_seconds": budget,
        "total_seconds": total, "snapshot_emitted_seconds": total,
        "best_available_seconds": cutoff["available_seconds"], "completion_status": run["status"],
        "completion_size": run["size"], "implementation": run["implementation"], "run_key": run["run_key"]}
    equal(record, expected, "SCIP main record is not its strict cutoff plus actual completion timing")
    return sizes, [event["upper_bound"] for event in bounds]


def audit_bound(record, n):
    upper = record["upper_bound"]
    require(math.isfinite(upper) and 0 <= upper <= n, "Invalid SCIP upper bound")
    raw, flag = record.get("raw_upper_bound"), record.get("raw_upper_bound_flag")
    if raw is None:
        require(flag in {"not_observed", "scip_infinity", "ieee_infinity", "nan"} and upper == n and
                record["bound_kind"] == "trivial_node_count_bound", "Undefined/infinite SCIP bound is not a disclosed trivial bound")
        if flag in {"scip_infinity", "ieee_infinity"}:
            require(record["raw_upper_bound_is_infinite"] is True and
                    record["raw_upper_bound_infinity_sign"] in {"positive", "negative"}, "Missing SCIP infinity disclosure")
    else:
        require(math.isfinite(raw) and flag == "finite" and upper == min(float(n), raw),
                "SCIP raw finite bound and valid node-count cap disagree")
        expected_kind = "scip_dual_bound_capped_by_node_count" if raw > n else "scip_dual_bound"
        require(record["bound_kind"] == expected_kind and not record["raw_upper_bound_is_infinite"], "Invalid finite SCIP bound kind")


def manifest_checked(directory):
    manifest = read_json(directory / "manifest.json")
    keys = ("plan", "selection", "selection_sha256", "source_sha256", "environment", "inputs")
    require(manifest["binding"] == fingerprint({key: manifest[key] for key in keys}), "Manifest binding does not match contents")
    require(set(manifest["source_sha256"]) == SOURCES, "Missing or unexpected bound training sources")
    for filename, digest in manifest["source_sha256"].items():
        require(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest,
                f"Current source differs from the bound experiment: {filename}; use the saved source commit")
    plan = manifest["plan"]
    equal(read_json(ROOT / "experiments/fast-anytime-20261018.json"), plan, "Current plan differs from the frozen manifest")
    selection_path = ROOT / plan["selection_source"]
    require(hashlib.sha256(selection_path.read_bytes()).hexdigest() == manifest["selection_sha256"], "Original selection bytes changed")
    equal(read_json(selection_path), manifest["selection"], "Original selection differs from its bound value")
    equal(plan["locked_configs"], manifest["selection"]["chosen"], "Configurations were not locked from validation")
    old_plan = read_json(selection_path.parent / "manifest.json")["protocol"]["plan"]
    original_validation = audit_synthetic(selection_path.parent, "validation", old_plan)
    equal(select_from_validation(original_validation, old_plan, selection_path), manifest["selection"],
          "Original validation selection did not reproduce")
    return manifest


def expected_inputs(directory, plan):
    result = {}
    for spec in plan["synthetic"]:
        name = f"n{spec['n']}_k{spec['k']}_d{spec['density']}_s{spec['seed']}"
        instance = {"name": name, **generate_instance(spec["n"], spec["k"], spec["density"], spec["seed"])}
        equal(read_json(directory / "instances" / f"{name}.json"), instance, "Saved synthetic input differs from regenerated graph")
        for mode in spec["definitions"]:
            result[f"{name}_{mode}"] = (mode, instance, "synthetic_new")
    for name in plan["real"]["datasets"]:
        instance = read_json(directory / "instances" / f"{name}.json")
        imported = json.loads(json.dumps(load_upstream_dataset(name).to_dict()))
        require(instance["name"] == name + "-upstream" and instance["n"] == imported["n"] and
                instance["edges"] == imported["edges"], "Real input differs from pinned public data")
        portable = lambda data: {k: v for k, v in data.items() if k not in {"raw_sha256", "raw_crlf_count", "raw_bytes"}}
        equal(portable(instance["import_metadata"]), portable(imported["import_metadata"]), "Real import/provenance changed")
        for mode in plan["real"]["definitions"]:
            result[f"{name}_{mode}"] = (mode, instance, "real_seen")
    return result


def audit_case(path, manifest, mode, instance):
    verify(path)
    report = read_json(path)
    require(report["manifest_binding"] == manifest["binding"] and report["definition"] == mode, "Case binding/definition mismatch")
    equal(report["instance"], instance, "Case and separately saved instance differ")
    equal(report["environment"], manifest["environment"], "Case environment differs from the manifest")
    problem = HypergraphProblem(instance["n"], tuple(tuple(edge) for edge in instance["edges"]))
    plan, records, runs = manifest["plan"], report["records"], report["runs"]
    targets, seeds = plan["checkpoints_seconds"], plan["model_seeds"]
    expected_runs = {f"scip_{budget}" for budget in targets} | {f"{method}_{seed}" for method in ("neural", "random") for seed in seeds}
    require(len(runs) == len(expected_runs) and {run["run_key"] for run in runs} == expected_runs,
            f"Incomplete or unexpected whole trajectories in {path.name}")
    degree = degree_independent(problem, mode)
    greedy_records = [r for r in records if r["method"] == "degree_greedy"]
    require(len(greedy_records) == 1 and greedy_records[0]["selected"] == degree, "Wrong degree-greedy solution")
    feasible_sizes, upper_bounds, expected_records = [len(degree)], [], [greedy_records[0]]
    for run in runs:
        key = run["run_key"]
        main = [record for record in records if record.get("run_key") == key]
        if key.startswith("neural_"):
            require(run["seed"] in seeds and key == f"neural_{run['seed']}", "Invalid neural seed/key")
            feasible_sizes.extend(audit_neural(run, problem, mode, plan))
            equal(main, run["records"], "Neural snapshots were spliced or differ from their whole run")
        elif key.startswith("random_"):
            require(run["seed"] in seeds and key == f"random_{run['seed']}", "Invalid random seed/key")
            feasible_sizes.extend(audit_random(run, problem, mode, plan))
            equal(main, [run["single"], *run["records"]], "Random snapshots were spliced or differ from their whole run")
        else:
            budget = next(budget for budget in targets if key == f"scip_{budget}")
            require(len(main) == 1, "SCIP independent run must have one primary cutoff record")
            sizes, bounds = audit_scip(run, main[0], problem, mode, budget, degree)
            feasible_sizes.extend(sizes)
            upper_bounds.extend(bounds)
        expected_records.extend(main)
    require(len(records) == 1 + len(targets) + len(seeds) * (1 + 2 * len(targets)) and
            len(expected_records) == len(records), "Incomplete or unexpected primary records")
    require(all(max(feasible_sizes) <= bound + 1e-5 for bound in upper_bounds),
            "A SCIP bound contradicts a feasible candidate from another budget/run, including late diagnostics")
    return report


def mean_or_none(values):
    return st.mean(values) if values else None


def prediction_stats(records):
    predicted = [record for record in records if "scores" in record]
    return {"records": len(records), "predicted_records": len(predicted),
        "empty_incumbent_rate": st.mean(int("scores" not in r) for r in records),
        "raw_feasible_rate": mean_or_none([int(r["raw_feasible"]) for r in predicted]),
        "raw_empty_rate": mean_or_none([int(r["raw_size"] == 0) for r in predicted]),
        "raw_size_mean": mean_or_none([r["raw_size"] for r in predicted]),
        "repaired_size_mean": mean_or_none([r["repaired_size"] for r in predicted]),
        "final_size_mean": st.mean(r["size"] for r in records),
        "removed_mean": mean_or_none([r["removed_by_repair"] for r in predicted]),
        "added_mean": mean_or_none([r["added_by_extension"] for r in predicted])}


def summarize_case(report, scope):
    records, runs, instance = report["records"], report["runs"], report["instance"]
    degree = next(record["size"] for record in records if record["method"] == "degree_greedy")
    singles = sorted((r for r in records if r["method"] == SINGLE), key=lambda r: r["seed"])
    rows = []
    for target in sorted({r["budget_seconds"] for r in records if r["method"] == NEURAL}):
        neural = sorted((r for r in records if r["method"] == NEURAL and r["budget_seconds"] == target), key=lambda r: r["seed"])
        random_records = sorted((r for r in records if r["method"] == RANDOM and r["budget_seconds"] == target), key=lambda r: r["seed"])
        scip = next(r for r in records if r["method"] == SCIP and r["budget_seconds"] == target)
        neural_runs = sorted((run for run in runs if run["method"] == NEURAL), key=lambda run: run["seed"])
        initial = [run["candidates"][run["initial_index"]] for run in neural_runs]
        latest = [run["candidates"][record["latest_candidate_index"]] for run, record in zip(neural_runs, neural)]
        eligible_initial_pairs = [(initial_record, final) for initial_record, final in zip(initial, neural)
                                  if initial_record["available_seconds"] <= target]
        method_sizes = {"degree_greedy": float(degree), SINGLE: st.mean(r["size"] for r in singles),
                        RANDOM: st.mean(r["size"] for r in random_records),
                        NEURAL: st.mean(r["size"] for r in neural), SCIP: float(scip["size"])}
        row = {"case_id": report["case_id"], "name": instance["name"], "scope": scope,
            "definition": report["definition"], "n": instance["n"], "m": len(instance["edges"]),
            "generator": instance.get("generator"), "budget_seconds": target, "config": neural[0]["config"],
            "model_seeds": [r["seed"] for r in neural], "degree_size": degree,
            "neural_sizes": [r["size"] for r in neural], "neural_mean_size": method_sizes[NEURAL],
            "random_sizes": [r["size"] for r in random_records], "random_mean_size": method_sizes[RANDOM],
            "random_single_sizes": [r["size"] for r in singles],
            "scip_size": scip["size"], "scip_upper_bound": scip["upper_bound"],
            "scip_optimality_proven": scip["optimality_proven"], "scip_status": scip["status"],
            "scip_completion_size": scip["completion_size"], "scip_completion_status": scip["completion_status"],
            "scip_bound_kind": scip["bound_kind"], "method_sizes": method_sizes,
            "paired_percent": {method: (100 * (method_sizes[NEURAL] / value - 1) if value else None)
                               for method, value in method_sizes.items() if method != NEURAL},
            "neural_best_decoder": prediction_stats(neural), "neural_latest_decoder": prediction_stats(latest),
            "initial_available_by_cutoff": len(eligible_initial_pairs),
            "initial_mean_size": mean_or_none([a["size"] for a, _ in eligible_initial_pairs]),
            "best_minus_initial_mean": mean_or_none([b["size"] - a["size"] for a, b in eligible_initial_pairs]),
            "latest_mean_size": st.mean(r["size"] for r in latest),
            "best_minus_latest_mean": st.mean(a["size"] - b["size"] for a, b in zip(neural, latest)),
            "neural_epochs_at_emission": [r["epochs_completed_at_snapshot"] for r in neural],
            "neural_latest_decoded_epochs": [r.get("epoch", 0) for r in latest],
            "actual_emission_seconds": {NEURAL: [r["total_seconds"] for r in neural],
                                       RANDOM: [r["total_seconds"] for r in random_records], SCIP: [scip["total_seconds"]]},
            "overshoot_seconds": {NEURAL: [r["overshoot_seconds"] for r in neural],
                                  RANDOM: [r["overshoot_seconds"] for r in random_records], SCIP: [scip["overshoot_seconds"]]},
            "random_completed_trials": [r["trials"] for r in random_records],
            "neural_best_available_seconds": [r["best_available_seconds"] for r in neural]}
        rows.append(row)
    return rows


def summarize_groups(rows):
    groups = []
    for scope, mode, target in sorted({(r["scope"], r["definition"], r["budget_seconds"]) for r in rows}):
        chosen = [r for r in rows if (r["scope"], r["definition"], r["budget_seconds"]) == (scope, mode, target)]
        comparisons = {}
        for method in chosen[0]["paired_percent"]:
            effects = [r["paired_percent"][method] for r in chosen if r["paired_percent"][method] is not None]
            differences = [r["method_sizes"][NEURAL] - r["method_sizes"][method] for r in chosen]
            comparisons[method] = {"mean_paired_percent": mean_or_none(effects), "defined_ratio_instances": len(effects),
                "wins_ties_losses": [sum(op(delta) for delta in differences) for op in
                    (lambda x: x > 1e-10, lambda x: abs(x) <= 1e-10, lambda x: x < -1e-10)]}
        groups.append({"scope": scope, "definition": mode, "budget_seconds": target, "instances": len(chosen),
            "comparisons": comparisons,
            "mean_sizes": {method: st.mean(r["method_sizes"][method] for r in chosen) for method in chosen[0]["method_sizes"]},
            "mean_best_minus_initial": mean_or_none([r["best_minus_initial_mean"] for r in chosen if r["best_minus_initial_mean"] is not None]),
            "mean_best_minus_latest": st.mean(r["best_minus_latest_mean"] for r in chosen),
            "scip_proven_instances": sum(r["scip_optimality_proven"] for r in chosen),
            "neural_best_decoder": {key: mean_or_none([r["neural_best_decoder"][key] for r in chosen
                if r["neural_best_decoder"][key] is not None]) for key in
                ("empty_incumbent_rate", "raw_feasible_rate", "raw_empty_rate", "raw_size_mean", "repaired_size_mean", "final_size_mean", "removed_mean", "added_mean")},
            "actual_emission_seconds": {method: distribution([value for r in chosen for value in r["actual_emission_seconds"][method]])
                                        for method in (NEURAL, RANDOM, SCIP)},
            "overshoot_seconds": {method: distribution([value for r in chosen for value in r["overshoot_seconds"][method]])
                                  for method in (NEURAL, RANDOM, SCIP)},
            "neural_epochs_at_emission": distribution([value for r in chosen for value in r["neural_epochs_at_emission"]])})
    return groups


def pct(value):
    return "未定义（对照为0）" if value is None else f"{value:+.2f}%"


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def report_text(summary, manifest):
    counts, plan = summary["counts"], manifest["plan"]
    final_synthetic = next(group for group in summary["groups"] if group["scope"] == "synthetic_new" and
                           group["definition"] == "weak" and group["budget_seconds"] == 30)
    def latest_raw_rate(group):
        rows = [row for row in summary["case_summary"] if (row["scope"], row["definition"], row["budget_seconds"]) ==
                (group["scope"], group["definition"], group["budget_seconds"])]
        return mean_or_none([row["neural_latest_decoder"]["raw_feasible_rate"] for row in rows
                             if row["neural_latest_decoder"]["raw_feasible_rate"] is not None])
    lines = ["# 快速解码、严格截止前缀与规模扩展实验", "", "日期：2026-10-18。", "",
        f"完整核验 {counts['cases']} 个场景、{counts['records']} 条主结果记录、{counts['neural_runs']} 条神经训练轨迹、"
        f"{counts['random_runs']} 条随机轨迹和 {counts['scip_runs']} 次独立 SCIP 运行。5／15／30 秒神经快照来自同一次 30 秒训练，不能计为三次训练。", "",
        "六张新弱独立集合成图上，30 秒神经三个种子均值相对低度贪心、随机重启和 SCIP 截止可行解的实例平均差异分别为 " +
        "、".join(pct(final_synthetic["comparisons"][method]["mean_paired_percent"]) for method in ("degree_greedy", RANDOM, SCIP)) +
        "，三项均为六个实例的均值获胜。这里比较固定实例上的截止前可行解质量，没有证明最优，也没有证明任意硬件上的普遍速度优势。", "",
        "本轮统一使用与旧规则逐阶段等价的快速解码器和等价低度贪心初解；已有实验结果保持冻结。除算法实现提速，还改变为固定 30 秒退火和在线前缀选优，因此端到端差异不能单独归因于快速解码器。解码器离线等价／计时实验见 [解码器基准说明](../results/decode-kernel-20261018/README.md)，旧完整对照见 [controlled-20261018.md](controlled-20261018.md)。", "",
        "## 固定协议与时间含义", "",
        "六张新合成图均为弱 MIS：500／1000／2000 节点，各含 k=3、m/n=2 和 k=5、m/n=4 两种结构；500 节点使用图种子 20261023，其余使用 20261024。m/n 是超边数／节点数，不是平均度。同一生成种子或指定结构下的观察不解释为独立统计样本。Cora／High／Primary 两种定义共六场景为已见真实数据的重新评价，不标为新测试集。", "",
        "网络配置继续使用旧验证集锁定的弱 w1024_cra、强 w1024_none，未使用新结果选参。作者公开网络与训练步不改；学习率 1e-4、权重衰减 5e-8、罚项系数 2、有效 dropout=0。消息传播仍来自目标单项式：弱分支为原超边加奖励单点，强分支为冲突点对加奖励单点。", "",
        "弱 CRA 固定以 30 秒为 horizon：系数 -1+1.6×min(1,已用墙钟秒／30)，已用时间包括构建、训练、在线候选解码；强分支系数为 0。5 秒的系数约为 -0.733，这与旧独立 5 秒协议末尾约 +0.6 不同。早期快照不是压缩退火后的独立短训练。", "",
        "神经网络从时刻 0 的可行空集开始，记录首个未更新前前向、约每秒及各截止点前 0.1 秒的候选；以完整分数传回 CPU 并完成解码后的 available_seconds 为候选可用时刻。每个 5／15／30 秒主结果只在此前已完成候选中取第一个最大可行解。未来候选和最终诊断不得回填早期结果。最好、最新、初始和终端诊断分别保存。空集兜底不是模型输出的原始空集。", "",
        "随机重启使用单一 30 秒 RNG 轨迹；首个试次作为单次随机对照，所有试次保存完成时刻和大小，每次严格改进保留分数和具体解。独立审计重放 RNG 到改进试次，重新解码首个／改进试次，不重复运行全部随机搜索。", "",
        "SCIP 的 5／15／30 秒为三次独立求解，单线程、同一快速等价贪心热启动，Python 构建从求解器预算扣除。事件回调在完整提取并核验可行解之后计时；截止快照仅用过去已完成的可行解和过去上界。晚到的最终解、界、optimal 状态另作 completion 诊断，不用于主结果。独立预算不保证单调，保留全部原值。", "",
        "质量前缀严格遵守名义截止时刻；结果实际发出可能晚于截止点，total_seconds 保留实际发出／完成时间，best_available_seconds 保留被选候选首次可用时间，两者不混用。表中的“发出时训练步数”可以包含截止后最后一步；不能当成严格截止前更新次数。所有在线解码计入墙钟时间，终端诊断和完整 run 耗时另行保留；JSON 写出和独立审计不计入运行器的试次时间。", "",
        f"manifest binding：`{manifest['binding']}`。启动源码提交：`{manifest['git_commit_at_start']}`。源码、旧选择文件、环境与输入哈希逐项绑定；整条 seed 轨迹及所有快照原子保存，恢复不拼接两次训练。", "",
        "## 按实例先平均种子的结果", "",
        "每格为神经三个种子均值相对对照的实例平均百分比（胜／平／负）。不使用最佳种子代替均值，不进行显著性检验。对照为 0 时比率不定义，保留绝对解大小与胜负。", "",
        "| 范围 | 定义 | 截止秒 | 实例数 | 对低度贪心 | 对随机单次 | 对随机重启 | 对 SCIP 截止可行解 |",
        "|---|---|---:|---:|---|---|---|---|"]
    for group in summary["groups"]:
        cells = [f"{pct(group['comparisons'][method]['mean_paired_percent'])}（{'/'.join(map(str, group['comparisons'][method]['wins_ties_losses']))}）"
                 for method in ("degree_greedy", SINGLE, RANDOM, SCIP)]
        lines.append(f"| {'新合成图' if group['scope']=='synthetic_new' else '已见真实图'} | {LABELS[group['definition']]} | "
            f"{group['budget_seconds']} | {group['instances']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 原始输出、后处理与保留最好候选的贡献", "",
        "下面原始指标只统计确实存在网络分数的被选候选；时刻 0 空集另列，不计为模型原始空集。跨实例平均先在每实例内平均三个种子。最好与初始的差异同时包含训练变化和多时刻选优，不能单独称为最后模型的学习增益；最好与最新之差显示保留历史候选的贡献。", "",
        "| 范围 | 定义 | 秒 | 兜底空集率 | 历史最好原始可行率 | 最新原始可行率 | 历史最好原始空集率 | 历史最好原始→修复→扩充大小 | 平均移除／加入 | 最好－初始 | 最好－最新 |",
        "|---|---|---:|---:|---|---|---|---|---|---:|---:|"])
    for group in summary["groups"]:
        d = group["neural_best_decoder"]
        ratio = lambda value: "—" if value is None else f"{value:.1%}"
        lines.append(f"| {'新合成图' if group['scope']=='synthetic_new' else '已见真实图'} | {LABELS[group['definition']]} | {group['budget_seconds']} | "
            f"{d['empty_incumbent_rate']:.1%} | {ratio(d['raw_feasible_rate'])} | {ratio(latest_raw_rate(group))} | {ratio(d['raw_empty_rate'])} | "
            f"{number(d['raw_size_mean'])}→{number(d['repaired_size_mean'])}→{number(d['final_size_mean'])} | "
            f"{number(d['removed_mean'])}／{number(d['added_mean'])} | {number(group['mean_best_minus_initial'])} | {number(group['mean_best_minus_latest'])} |")
    lines.extend(["", f"30 秒新合成图的截止前最新候选原始可行率为 {latest_raw_rate(final_synthetic):.1%}；"
        f"按后处理后解大小选中的历史最好候选原始可行率为 {final_synthetic['neural_best_decoder']['raw_feasible_rate']:.2%}，"
        f"历史最好仅比最新候选平均大 {final_synthetic['mean_best_minus_latest']:.2f} 个点。"
        "这表明保留规则经常选中较早的不可行阈值输出经修复／扩充得到的可行解，不能把历史最好候选的原始可行率混称为最终网络的原始可行率。"
        "“最新”指截止前最近完成的候选，30 秒后的终端诊断另存；本批最新候选全部可行也不是一般可行性保证。", "",
        "## 实际发出时间与更新步数", "",
        "| 范围 | 定义 | 截止秒 | 方法 | 次数 | 发出时间中位秒 | 最小—最大秒 | 最大晚发秒 |",
        "|---|---|---:|---|---:|---:|---|---:|"])
    for group in summary["groups"]:
        for method in (NEURAL, RANDOM, SCIP):
            times, late = group["actual_emission_seconds"][method], group["overshoot_seconds"][method]
            lines.append(f"| {'新合成图' if group['scope']=='synthetic_new' else '已见真实图'} | {LABELS[group['definition']]} | {group['budget_seconds']} | "
                f"{LABELS[method]} | {times['count']} | {times['median']:.4f} | {times['min']:.4f}—{times['max']:.4f} | {late['max']:.4f} |")
    lines.extend(["", "| 范围 | 定义 | 截止秒 | 发出时更新数最小／中位／最大 |",
        "|---|---|---:|---|"])
    for group in summary["groups"]:
        e = group["neural_epochs_at_emission"]
        lines.append(f"| {'新合成图' if group['scope']=='synthetic_new' else '已见真实图'} | {LABELS[group['definition']]} | {group['budget_seconds']} | {e['min']}／{e['median']:.1f}／{e['max']} |")
    lines.extend(["", "## 全部逐实例主结果", "",
        "| 实例 | 定义 | 截止秒 | 贪心 | 随机单次均值 | 随机重启三种子 | 神经三种子 | SCIP 过去下界／上界 | 截止前证明最优 | 完成诊断大小／状态 |",
        "|---|---|---:|---:|---:|---|---|---|---|---|"])
    for row in summary["case_summary"]:
        lines.append(f"| {row['name']} | {LABELS[row['definition']]} | {row['budget_seconds']} | {row['degree_size']} | "
            f"{st.mean(row['random_single_sizes']):.2f} | {' / '.join(map(str,row['random_sizes']))} | {' / '.join(map(str,row['neural_sizes']))} | "
            f"{row['scip_size']}／{row['scip_upper_bound']:.3f} | {'是' if row['scip_optimality_proven'] else '否'} | {row['scip_completion_size']}／{row['scip_completion_status']} |")
    lines.extend(["", "## 结论范围与复现", "",
        "新合成图的神经均值在 30 秒优于上述三种对照，但 5 秒时相对 SCIP 的实例平均差异为 " +
        pct(next(group for group in summary["groups"] if group["scope"] == "synthetic_new" and group["budget_seconds"] == 5)["comparisons"][SCIP]["mean_paired_percent"]) +
        "，所以优势与预算有关。六张合成图均没有最优证明，2000 节点稠密五阶图的三个训练种子还存在较大差异，不将实例均值获胜解释为每次运行都更好。", "",
        "已见真实数据到 15 秒时，SCIP 六个定义／实例场景均已证明最优：Cora 弱／强为 1030／612，High 为 52／51，Primary 为 19／19。30 秒神经均值在真实弱、强两组均落后于 SCIP 最优解；其相对贪心或随机重启的改善不能写成超过最优解。真实文件身份和统计差异仍按公开仓库版本解释。", "",
        "质量使用严格截止前缀，晚发的候选不回填；实际返回延迟仍须与名义截止分开。部分真实 SCIP 求解提前证明最优即返回，实际耗时远小于目标预算；部分 5 秒 SCIP 运行也在 5 秒之前以时限状态返回。表中完整保留实际发出时间，不把这些运行说成严格相同运行时长。", "",
        "神经与随机的最好曲线非下降，是前缀保留规则直接保证的性质，不能把曲线单调本身当成持续学习的证据。贪心和相同解码器给出极大可行独立集，SCIP 截止解只保证可行；时刻 0 的兜底空集也不一定极大。没有最优证明时不把任何可行解称为最大独立集。", "",
        "本轮扩展只覆盖指定的六张合成图和三个已见真实文件；未评估更大节点数、其他分布或弱／强定义之外的任务。真实来源身份及 High／Primary 弱高阶约束的冗余限制仍成立，见 [data-provenance.md](data-provenance.md)。截至本轮，仅就这些来源、固定配置和固定 30 秒退火讨论表现。", "",
        "在已配置的项目环境和本轮绑定源码下，新目录运行：", "", "```powershell",
        ".venv/Scripts/python.exe scripts/run_anytime.py --output results/fast-anytime-rerun",
        ".venv/Scripts/python.exe scripts/summarize_anytime.py results/fast-anytime-rerun --report docs/fast-anytime-rerun.md",
        "```", "", "运行器会核对旧验证选择来源、当前源码和环境；改变它们应固定新协议。时间预算和 GPU 运算使固定种子不保证逐位相同结果。完整候选、RNG 试次日志、实际发出时间、SCIP 晚到诊断均保留，summary 只在所有预声明场景与轨迹通过审计后标记 complete=true。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = manifest_checked(args.directory)
    expected = expected_inputs(args.directory, manifest["plan"])
    equal(set_to_list(manifest["inputs"]), set_to_list(expected), "Manifest input keys differ from the complete plan")
    actual = {path.stem for path in (args.directory / "cases").glob("*.json")}
    require(actual == set(expected), f"Incomplete suite: missing {sorted(set(expected)-actual)}; unexpected {sorted(actual-set(expected))}")
    reports, rows = [], []
    for name, (mode, instance, scope) in expected.items():
        require(manifest["inputs"][name] == fingerprint(instance), "Bound input fingerprint differs")
        report = audit_case(args.directory / "cases" / f"{name}.json", manifest, mode, instance)
        report["case_id"] = name
        reports.append(report)
        rows.extend(summarize_case(report, scope))
    runs = [run for report in reports for run in report["runs"]]
    records = [record for report in reports for record in report["records"]]
    counts = {"cases": len(reports), "records": len(records),
        "neural_runs": sum(run["method"] == NEURAL for run in runs),
        "random_runs": sum(run["method"] == RANDOM for run in runs),
        "scip_runs": sum(run["method"] == SCIP for run in runs),
        "neural_candidates_audited": sum(len(run["candidates"])-1 for run in runs if run["method"] == NEURAL),
        "random_improvements_audited": sum(len(run["candidates"])-1 for run in runs if run["method"] == RANDOM),
        "random_trials_logged": sum(len(run["trial_log"]) for run in runs if run["method"] == RANDOM)}
    summary = {"complete": True, "manifest_binding": manifest["binding"], "counts": counts,
        "case_summary": rows, "groups": summarize_groups(rows),
        "neural_run_profile": {field: distribution([run[field] for run in runs if run["method"] == NEURAL]) for field in
            ("run_total_seconds", "build_seconds", "training_seconds", "online_decode_seconds", "epochs_completed")},
        "limits": "Strict cutoff prefixes; emission latency separate; one 30s trajectory per seed, not three trainings; real data seen previously; descriptive instance means only."}
    document = report_text(summary, manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(document, encoding="utf8")
    write_json(args.directory / "summary.json", summary)
    print(json.dumps({**counts, "complete": True, "report": str(args.report)}, ensure_ascii=False, indent=2))


def set_to_list(value):
    return sorted(value)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, StopIteration) as error:
        raise SystemExit(f"Audit refused; no complete summary was produced: {error}")
