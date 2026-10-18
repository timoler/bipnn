"""Require complete, bound validation/test suites and audit saved solution stages.

No model is trained by this script. Repeated seeds are averaged within each
instance before comparisons across instances; all effects are descriptive.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mis.benchmark import generate_instance, write_json
from mis.problem import HypergraphProblem, degree_greedy
from mis.verify_report import verify

NEURAL = "public_architecture_indexed_budget"
CONFIGS = {
    "w256_cra": {"width": 256, "annealing": "cra"},
    "w256_none": {"width": 256, "annealing": "none"},
    "w1024_cra": {"width": 1024, "annealing": "cra"},
    "w1024_none": {"width": 1024, "annealing": "none"},
}
LABELS = {
    "weak": "弱独立集", "strong": "强独立集", NEURAL: "神经网络＋同一解码器",
    "degree_greedy": "低度贪心", "random_score_same_decoder": "随机分数单次解码",
    "random_restart_same_decoder": "随机分数重启", "scip_milp_greedy_start": "SCIP＋低度贪心初解",
}


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid {name}: {value!r}")


def distribution(values):
    values = sorted(values)
    require(bool(values), "Cannot summarize an empty distribution")
    position = 0.9 * (len(values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return {"count": len(values), "mean": st.mean(values), "median": st.median(values),
            "min": values[0], "p90": values[lo] + (values[hi] - values[lo]) * (position - lo),
            "max": values[-1]}


def solution(problem, mode, record, key="selected", size_key="size", feasible=True):
    selected = record[key]
    require(isinstance(selected, list) and len(selected) == len(set(selected)), f"Duplicate {key}")
    require(all(type(v) is int and 0 <= v < problem.n for v in selected), f"Invalid {key}")
    require(type(record[size_key]) is int and len(selected) == record[size_key], f"Wrong {size_key}")
    actual = problem.feasible(selected, mode)
    if feasible:
        require(actual, f"Infeasible {key}")
    return actual


def audit_decoder(problem, mode, record, full_stages):
    """Reimplement the saved deterministic threshold/repair/extension policy.

    This deliberately does not call decode_stages or repair_and_extend: it
    verifies the probabilities and each saved intermediate set independently.
    """
    scores = record["scores"]
    require(len(scores) == problem.n and all(type(p) in (int, float) and
            math.isfinite(p) and 0 <= p <= 1 for p in scores), "Invalid sigmoid scores")
    raw = [v for v, p in enumerate(scores) if p >= 0.5]
    require(record["raw_selected"] == raw, "Raw set differs from thresholded probabilities")
    actual = solution(problem, mode, record, "raw_selected", "raw_size", feasible=False)
    require(type(record["raw_feasible"]) is bool and record["raw_feasible"] == actual,
            "Incorrect raw feasibility flag")
    selected = set(raw)
    incidence = [[] for _ in range(problem.n)]
    for edge_index, edge in enumerate(problem.edges):
        for v in edge:
            incidence[v].append(edge_index)
    caps = [len(edge) - 1 if mode == "weak" else 1 for edge in problem.edges]
    counts = [sum(v in selected for v in edge) for edge in problem.edges]
    bad = {index for index, count in enumerate(counts) if count > caps[index]}
    involved = {v for index in bad for v in problem.edges[index] if v in selected}
    # Removing a selected vertex cannot introduce a new violated constraint.
    # Sorting the original candidate pool once therefore reproduces repeated
    # lowest-score removal, while remaining fast on dense real-data files.
    for v in sorted(involved, key=lambda v: (scores[v], v)):
        if not bad:
            break
        if any(index in bad for index in incidence[v]):
            selected.remove(v)
            for index in incidence[v]:
                counts[index] -= 1
                if counts[index] <= caps[index]:
                    bad.discard(index)
    require(not bad, "Independent repair did not resolve all conflicts")
    repaired = sorted(selected)
    if full_stages:
        solution(problem, mode, record, "repaired_selected", "repaired_size")
        require(record["repaired_selected"] == repaired, "Incorrect intermediate repaired set")
        require(record["removed_by_repair"] == len(raw) - len(repaired), "Wrong repair removal count")
    for v in sorted(range(problem.n), key=lambda v: (-scores[v], v)):
        if v not in selected and all(counts[index] < caps[index] for index in incidence[v]):
            selected.add(v)
            for index in incidence[v]:
                counts[index] += 1
    require(record["selected"] == sorted(selected), "Final set differs from the declared decoder")
    if full_stages:
        require(record["added_by_extension"] == len(selected) - len(repaired), "Wrong extension addition count")
    solution(problem, mode, record)
    require(record["feasible"] is True, "Final feasibility flag must be true")


def audit_neural(problem, mode, record, budget):
    config = record["config"]
    require(config in CONFIGS, "Unknown neural configuration")
    settings = record["settings"]
    required = {**CONFIGS[config], "lr": 1e-4, "weight_decay": 5e-8,
                "penalty": 2.0, "effective_dropout": 0.0,
                "budget_seconds": budget, "fixed_epochs": None,
                "cra_schedule": "-1+1.6*(elapsed setup+train / budget)",
                "output": "last training-mode forward before final update"}
    require(settings == required, "Neural settings differ from the declared timed experiment")
    require(type(record["epochs_completed"]) is int and record["epochs_completed"] >= 1,
            "Invalid completed training step count")
    for key in ("build_seconds", "training_seconds", "decode_seconds", "diagnostic_decode_seconds"):
        finite_nonnegative(record[key], key)
    require(math.isclose(record["total_seconds"], sum(record[k] for k in
            ("build_seconds", "training_seconds", "decode_seconds")), rel_tol=1e-9, abs_tol=1e-8),
            "Primary time does not equal build + training + final decoding")
    audit_decoder(problem, mode, record, full_stages=True)
    audit_decoder(problem, mode, record["initial"], full_stages=True)
    previous_epoch, previous_seconds = 0, -1.0
    for checkpoint in record["checkpoints"]:
        epoch, seconds = checkpoint["epoch"], checkpoint["seconds"]
        require(type(epoch) is int and previous_epoch < epoch <= record["epochs_completed"],
                "Nonmonotone checkpoint epochs")
        finite_nonnegative(seconds, "checkpoint seconds")
        require(previous_seconds <= seconds <= record["total_seconds"], "Invalid checkpoint time")
        require(math.isfinite(checkpoint["loss_before_update"]), "Nonfinite checkpoint loss")
        coefficient = checkpoint["annealing_coefficient"]
        require(math.isfinite(coefficient) and (-1 <= coefficient <= 0.6 + 1e-12
                if CONFIGS[config]["annealing"] == "cra" else coefficient == 0),
                "Invalid annealing coefficient")
        audit_decoder(problem, mode, checkpoint, full_stages=True)
        previous_epoch, previous_seconds = epoch, seconds
    require(bool(record["checkpoints"]), "Missing training checkpoints")
    return len(record["checkpoints"]) + 2


def identity(record):
    return record["method"], record.get("config"), record.get("seed")


def expected_records(phase, settings, mode, selection):
    required = [("degree_greedy", None, None)]
    configs = settings["configs"] if phase == "validation" else [selection["chosen"][mode]]
    if phase != "validation":
        required.append(("scip_milp_greedy_start", None, None))
        for seed in settings["model_seeds"]:
            required.extend([("random_score_same_decoder", None, seed),
                             ("random_restart_same_decoder", None, seed)])
    required.extend((NEURAL, config, seed) for config in configs for seed in settings["model_seeds"])
    return set(required)


def bound_manifest(directory, phase, plan, selection=None):
    path = directory / "manifest.json"
    require(path.exists(), f"Incomplete {phase}: missing manifest {path}")
    manifest = read_json(path)
    protocol = manifest["protocol"]
    require(protocol["phase"] == phase and protocol["plan"] == plan, f"Wrong {phase} protocol")
    require(protocol.get("selection") == selection, f"Wrong locked selection in {phase} manifest")
    require(manifest["binding"] == fingerprint({"protocol": protocol,
            "source_sha256": manifest["source_sha256"]}), f"Incorrect {phase} manifest binding")
    required_sources = {"mis/controlled.py", "mis/indexed_loss.py", "mis/problem.py",
                        "mis/benchmark.py", "mis/reference.py", "src/models.py", "src/train.py",
                        "scripts/run_controlled.py"}
    require(set(manifest["source_sha256"]) == required_sources, "Missing or unexpected bound source files")
    require(all(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                for digest in manifest["source_sha256"].values()), "Invalid bound source SHA-256")
    return manifest


def synthetic_instances(settings, definitions):
    result = {}
    for n, k, density, seed in itertools.product(settings["sizes"], settings["orders"],
                                                settings["densities_m_over_n"], settings["instance_seeds"]):
        name = f"n{n}_k{k}_d{density}_s{seed}"
        instance = {"name": name, **generate_instance(n, k, density, seed)}
        for mode in definitions:
            result[f"{name}_{mode}.json"] = (mode, instance)
    return result


def audit_case(path, manifest, mode, instance, required, budget):
    verify(path)
    report = read_json(path)
    require(report["manifest_binding"] == manifest["binding"], f"Unbound case: {path.name}")
    require(report["definition"] == mode, f"Wrong definition: {path.name}")
    require(fingerprint(report["instance"]) == fingerprint(instance), f"Wrong saved instance: {path.name}")
    require(report["environment"] == manifest["environment"], f"Wrong environment: {path.name}")
    ids = [identity(record) for record in report["records"]]
    require(len(ids) == len(set(ids)) and set(ids) == required,
            f"Incomplete or unexpected method/config/seed records: {path.name}")
    problem = HypergraphProblem(instance["n"], tuple(tuple(edge) for edge in instance["edges"]))
    checkpoint_sets = 0
    degree = next(r for r in report["records"] if r["method"] == "degree_greedy")
    require(degree["selected"] == degree_greedy(problem, mode), "Greedy result differs from declared algorithm")
    feasible_sizes = [r["size"] for r in report["records"]]
    for record in report["records"]:
        if record["method"] == NEURAL:
            feasible_sizes.append(record["initial"]["size"])
            feasible_sizes.extend(checkpoint["size"] for checkpoint in record["checkpoints"])
    best = max(feasible_sizes)
    for record in report["records"]:
        solution(problem, mode, record)
        finite_nonnegative(record["total_seconds"], "total_seconds")
        method = record["method"]
        if method == NEURAL:
            checkpoint_sets += audit_neural(problem, mode, record, budget)
        elif method.startswith("random_"):
            audit_decoder(problem, mode, record, full_stages=False)
            require(type(record["trials"]) is int and record["trials"] >= 1, "Invalid random trial count")
            limit = budget if method == "random_restart_same_decoder" else None
            require(record["time_limit_seconds"] == limit, "Incorrect random baseline budget")
            if limit is None:
                require(record["trials"] == 1, "Single-decode baseline used multiple trials")
        elif method == "scip_milp_greedy_start":
            require(record["time_limit_seconds"] == budget and record["threads"] == 1,
                    "SCIP configuration differs from the declared experiment")
            require(record["warm_start"] == "degree_greedy" and record["warm_start_size"] == degree["size"],
                    "SCIP warm-start mismatch")
            for key in ("build_seconds", "solver_seconds", "solver_time_limit_seconds", "mip_gap"):
                finite_nonnegative(record[key], key)
            require(math.isclose(record["solver_time_limit_seconds"],
                    max(0.001, budget - record["build_seconds"]), rel_tol=0, abs_tol=1e-9),
                    "SCIP budget did not subtract model and warm-start construction")
            upper = record["upper_bound"]
            require(math.isfinite(upper) and best <= upper + 1e-5 and upper <= problem.n + 1e-5,
                    "SCIP upper bound contradicts a saved feasible solution")
            require(record["size"] >= degree["size"], "SCIP lost its feasible warm start")
            require(record["optimality_proven"] == (record["status"] == "optimal"), "Invalid optimality label")
            if record["optimality_proven"]:
                require(abs(upper - record["size"]) <= 1e-5, "SCIP optimum and bound disagree")
    return report, checkpoint_sets


def audit_synthetic(directory, phase, plan, selection=None):
    settings = plan[phase]
    manifest = bound_manifest(directory, phase, plan, selection)
    expected = synthetic_instances(settings, plan["definitions"])
    actual = {path.name for path in (directory / "cases").glob("*.json")}
    missing, extra = sorted(set(expected) - actual), sorted(actual - set(expected))
    require(not missing and not extra, f"Incomplete {phase}: missing {missing}; unexpected {extra}")
    reports, checkpoint_sets = [], 0
    for filename, (mode, instance) in expected.items():
        report, count = audit_case(directory / "cases" / filename, manifest, mode, instance,
                                   expected_records(phase, settings, mode, selection), plan["budget_seconds"])
        require(report["purpose"] == phase, f"Wrong phase label in {filename}")
        reports.append(report)
        checkpoint_sets += count
    return {"manifest": manifest, "reports": reports, "audited_neural_stage_sets": checkpoint_sets}


def audit_real(directory, plan, selection):
    from mis.datasets import load_upstream_dataset
    manifest = bound_manifest(directory, "real", plan, selection)
    protocol = manifest["protocol"]
    require(set(protocol["datasets"]) == {"cora", "high", "primary"}, "Incomplete real dataset selection")
    require(protocol["model_seeds"] == plan["test"]["model_seeds"] and
            protocol["definitions"] == plan["definitions"], "Real seeds/definitions differ from locked evaluation")
    expected = {f"{name}_{mode}.json" for name in protocol["datasets"] for mode in protocol["definitions"]}
    actual = {path.name for path in (directory / "cases").glob("*.json")}
    require(actual == expected, f"Incomplete real data: missing {sorted(expected - actual)}; unexpected {sorted(actual - expected)}")
    reports, checkpoint_sets = [], 0
    for name, digest in protocol["datasets"].items():
        instance = read_json(directory / "instances" / f"{name}-upstream.json")
        without_name = {key: value for key, value in instance.items() if key != "name"}
        require(instance["name"] == name + "-upstream" and fingerprint(without_name) == digest,
                "Real instance does not match its bound pre-name fingerprint")
        imported = json.loads(json.dumps(load_upstream_dataset(name).to_dict()))
        require(instance["n"] == imported["n"] and instance["edges"] == imported["edges"],
                "Saved real graph differs from the pinned public source")
        # Git may check out the same pinned text as CRLF or LF. Dataset identity
        # is the LF-normalized hash; retain the original run's raw-byte audit.
        portable = lambda metadata: {key: value for key, value in metadata.items()
                                     if key not in {"raw_sha256", "raw_crlf_count", "raw_bytes"}}
        require(portable(instance["import_metadata"]) == portable(imported["import_metadata"]),
                "Real-data provenance or preprocessing metadata differs from the pinned import policy")
        raw_hash = instance["import_metadata"]["raw_sha256"]
        require(len(raw_hash) == 64 and all(c in "0123456789abcdef" for c in raw_hash), "Invalid saved raw data hash")
        for mode in protocol["definitions"]:
            report, count = audit_case(directory / "cases" / f"{name}_{mode}.json", manifest, mode, instance,
                                       expected_records("real", protocol, mode, selection), plan["budget_seconds"])
            require(report["purpose"] == "real", "Real-data record mislabeled")
            reports.append(report)
            checkpoint_sets += count
    return {"manifest": manifest, "reports": reports, "audited_neural_stage_sets": checkpoint_sets}


def select_from_validation(suite, plan, selection_path):
    scores = {mode: {} for mode in plan["definitions"]}
    for mode in scores:
        reports = [r for r in suite["reports"] if r["definition"] == mode]
        for config in plan["validation"]["configs"]:
            ratios = []
            for report in reports:
                degree = next(r["size"] for r in report["records"] if r["method"] == "degree_greedy")
                values = [r["size"] for r in report["records"] if r.get("config") == config]
                ratios.append(st.mean(values) / degree)
            scores[mode][config] = st.mean(ratios)
    chosen = {}
    for mode, values in scores.items():
        top = max(values.values())
        tied = [c for c in values if top - values[c] <= 1e-12]
        chosen[mode] = min(tied, key=lambda c: (CONFIGS[c]["width"], CONFIGS[c]["annealing"] != "cra"))
    require(selection_path.exists(), f"Validation selection is not locked: {selection_path}")
    selection = read_json(selection_path)
    require(selection["plan_sha256"] == fingerprint(plan) and
            selection["validation_binding"] == suite["manifest"]["binding"] and
            selection["criterion"] == plan["selection"] and selection["chosen"] == chosen,
            "Saved selection differs from the predeclared validation rule")
    require(set(selection["validation_scores"]) == set(scores), "Wrong validation score definitions")
    for mode, values in scores.items():
        require(set(selection["validation_scores"][mode]) == set(values), "Missing validation configurations")
        for config, value in values.items():
            require(math.isclose(selection["validation_scores"][mode][config], value,
                                 rel_tol=0, abs_tol=1e-12), "Saved validation score was not reproduced")
    return selection


def decoder_stats(records):
    return {"runs": len(records),
            "raw_feasible_rate": st.mean(int(r["raw_feasible"]) for r in records),
            "raw_empty_rate": st.mean(int(r["raw_size"] == 0) for r in records),
            "raw_size_mean": st.mean(r["raw_size"] for r in records),
            "repaired_size_mean": st.mean(r["repaired_size"] for r in records),
            "final_size_mean": st.mean(r["size"] for r in records),
            "removed_by_repair_mean": st.mean(r["removed_by_repair"] for r in records),
            "added_by_extension_mean": st.mean(r["added_by_extension"] for r in records),
            "initial_final_size_mean": st.mean(r["initial"]["size"] for r in records),
            "trained_minus_initial_mean": st.mean(r["size"] - r["initial"]["size"] for r in records),
            "trained_vs_initial_wins_ties_losses": [sum(1 for r in records if op(r["size"], r["initial"]["size"]))
                for op in (lambda a, b: a > b, lambda a, b: a == b, lambda a, b: a < b)],
            "initial_raw_feasible_rate": st.mean(int(r["initial"]["raw_feasible"]) for r in records),
            "total_seconds": distribution([r["total_seconds"] for r in records]),
            "diagnostic_decode_seconds": distribution([r["diagnostic_decode_seconds"] for r in records]),
            "epochs_completed": distribution([r["epochs_completed"] for r in records])}


def case_rows(suite):
    rows = []
    for report in suite["reports"]:
        instance = report["instance"]
        for config in sorted({r["config"] for r in report["records"] if r["method"] == NEURAL}):
            grouped = {}
            for record in report["records"]:
                if record["method"] != NEURAL or record["config"] == config:
                    grouped.setdefault(record["method"], []).append(record)
            means = {method: st.mean(r["size"] for r in records) for method, records in grouped.items()}
            neural = grouped[NEURAL]
            scip = grouped.get("scip_milp_greedy_start", [None])[0]
            row = {"name": instance["name"], "definition": report["definition"], "config": config,
                   "n": instance["n"], "m": len(instance["edges"]), "generator": instance.get("generator"),
                   "method_mean_sizes": means,
                   "neural_sizes": [r["size"] for r in neural],
                   "neural_seeds": [r["seed"] for r in neural], "decoder": decoder_stats(neural),
                   "method_actual_seconds": {method: distribution([r["total_seconds"] for r in records])
                                             for method, records in grouped.items()},
                   "paired_percent": {method: 100 * (means[NEURAL] / mean - 1)
                                      for method, mean in means.items() if method != NEURAL},
                   "scip": None if scip is None else {key: scip[key] for key in
                       ("size", "upper_bound", "mip_gap", "status", "optimality_proven", "total_seconds",
                        "build_seconds", "solver_seconds", "solver_time_limit_seconds", "warm_start_size")}}
            rows.append(row)
    return rows


def aggregate(rows):
    result = []
    keys = sorted({(row["definition"], row["config"]) for row in rows})
    for mode, config in keys:
        selected = [row for row in rows if (row["definition"], row["config"]) == (mode, config)]
        comparisons = {}
        for baseline in selected[0]["paired_percent"]:
            effects = [row["paired_percent"][baseline] for row in selected]
            comparisons[baseline] = {"mean_paired_percent": st.mean(effects),
                "wins_ties_losses": [sum(op(effect, 0) for effect in effects) for op in
                    (lambda a, b: a > b + 1e-10, lambda a, b: abs(a - b) <= 1e-10,
                     lambda a, b: a < b - 1e-10)]}
        decoder_keys = [key for key in selected[0]["decoder"] if key.endswith("_mean") or key.endswith("_rate")]
        result.append({"definition": mode, "config": config, "instances": len(selected),
                       "mean_sizes": {method: st.mean(row["method_mean_sizes"][method] for row in selected)
                                      for method in selected[0]["method_mean_sizes"]},
                       "decoder": {key: st.mean(row["decoder"][key] for row in selected) for key in decoder_keys},
                       "comparisons": comparisons})
    return result


def suite_summary(suite):
    rows = case_rows(suite)
    records = [record for report in suite["reports"] for record in report["records"]]
    neural_profile = []
    for mode, config in sorted({(report["definition"], record["config"])
            for report in suite["reports"] for record in report["records"] if record["method"] == NEURAL}):
        runs = [record for report in suite["reports"] if report["definition"] == mode
                for record in report["records"] if record["method"] == NEURAL and record["config"] == config]
        neural_profile.append({"definition": mode, "config": config, "runs": len(runs),
            "epochs_completed": distribution([r["epochs_completed"] for r in runs]),
            "stage_seconds": {stage: distribution([r[stage] for r in runs]) for stage in
                ("build_seconds", "training_seconds", "decode_seconds", "diagnostic_decode_seconds")}})
    return {"complete": True, "manifest_binding": suite["manifest"]["binding"],
            "cases": len(suite["reports"]), "records": len(records),
            "training_runs": sum(r["method"] == NEURAL for r in records),
            "audited_neural_stage_sets": suite["audited_neural_stage_sets"],
            "method_records": dict(Counter(r["method"] for r in records)),
            "aggregate": aggregate(rows), "instance_results": rows, "neural_profile": neural_profile,
            "timing_all_runs": {method + (":" + config if config else ""):
                distribution([r["total_seconds"] for r in records if r["method"] == method and r.get("config") == config])
                for method, config in sorted({(r["method"], r.get("config")) for r in records}, key=str)},
            "note": "Seeds averaged within each instance before paired effects; descriptive only; density means m/n."}


def percent(value):
    return f"{value:+.2f}%"


def append_results(lines, title, summary):
    lines.extend([f"## {title}", "", f"完整核验 {summary['cases']} 个定义／实例场景、{summary['records']} 条最终解记录、"
                  f"{summary['training_runs']} 次训练，以及 {summary['audited_neural_stage_sets']} 组神经输出阶段证据。", "",
                  "| 定义 | 配置 | 实例数 | 相对低度贪心 | 相对随机单次 | 相对随机重启 | 相对 SCIP 可行解 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in summary["aggregate"]:
        effects = row["comparisons"]
        cells = [f"{percent(effects[m]['mean_paired_percent'])}（{'/'.join(map(str, effects[m]['wins_ties_losses']))}）"
                 if m in effects else "—" for m in ("degree_greedy", "random_score_same_decoder",
                                                      "random_restart_same_decoder", "scip_milp_greedy_start")]
        lines.append(f"| {LABELS[row['definition']]} | {row['config']} | {row['instances']} | " + " | ".join(cells) + " |")
    lines.extend(["", "单元格为平均百分比差异（按实例的胜／平／负）。百分比先在每张实例上计算“神经网络三个（验证集为两个）种子平均解大小／该对照的种子平均解大小－1”，再对实例平均；未使用三个种子的最好解代替平均解。SCIP 栏比较截止时已找到的最好可行解，只有标为 optimal 的场景才已证明最优。", "",
                  "| 定义 | 配置 | 原始可行率 | 原始空集率 | 原始大小→修复后→扩充后 | 修复平均移除 | 扩充平均加入 | 未训练同解码器→训练后 |",
                  "|---|---|---:|---:|---|---:|---:|---|"])
    for row in summary["aggregate"]:
        d = row["decoder"]
        lines.append(f"| {LABELS[row['definition']]} | {row['config']} | {d['raw_feasible_rate']:.1%} | "
                     f"{d['raw_empty_rate']:.1%} | {d['raw_size_mean']:.2f}→{d['repaired_size_mean']:.2f}→{d['final_size_mean']:.2f} | "
                     f"{d['removed_by_repair_mean']:.2f} | {d['added_by_extension_mean']:.2f} | "
                     f"{d['initial_final_size_mean']:.2f}→{d['final_size_mean']:.2f} |")
    lines.extend(["", "原始可行率仅描述本批保存的阈值输出；最终可行性来自显式修复和扩充，不是神经网络的一般保证。初始对照为同一随机种子、同一模型第一步参数更新前的训练态前向，采用完全相同的解码器；两者差异用于量化此次训练是否改善输出排序。", "",
                  "| 方法／配置 | 次数 | 平均秒 | 中位秒 | P90 秒 | 最小—最大秒 |",
                  "|---|---:|---:|---:|---:|---|"])
    for method, times in summary["timing_all_runs"].items():
        name, _, config = method.partition(":")
        label = LABELS.get(name, name) + (" / " + config if config else "")
        lines.append(f"| {label} | {times['count']} | {times['mean']:.3f} | {times['median']:.3f} | "
                     f"{times['p90']:.3f} | {times['min']:.3f}—{times['max']:.3f} |")
    lines.extend(["", "实际耗时包含模型／约束构建和主解码；神经网络初始与轨迹解码在主结果计时结束后执行，单独保留 diagnostic_decode_seconds。P90 为线性插值描述统计。桌面环境未隔离其他负载，不能据此声称速度优势。", ""])
    lines.extend(["| 定义 | 配置 | 训练次数 | 完成步数最小 | 中位 | 最大 | 平均 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in summary["neural_profile"]:
        epochs = row["epochs_completed"]
        lines.append(f"| {LABELS[row['definition']]} | {row['config']} | {row['runs']} | {epochs['min']} | "
                     f"{epochs['median']:.1f} | {epochs['max']} | {epochs['mean']:.1f} |")
    lines.extend(["", "| 定义 | 配置 | 时间组成 | 次数 | 平均秒 | 中位秒 | P90 秒 | 最小—最大秒 |",
                  "|---|---|---|---:|---:|---:|---:|---|"])
    stage_labels = {"build_seconds": "模型和目标构建", "training_seconds": "训练",
                    "decode_seconds": "最终主解码", "diagnostic_decode_seconds": "初始／轨迹诊断解码（额外）"}
    for row in summary["neural_profile"]:
        for stage, times in row["stage_seconds"].items():
            lines.append(f"| {LABELS[row['definition']]} | {row['config']} | {stage_labels[stage]} | {times['count']} | "
                         f"{times['mean']:.3f} | {times['median']:.3f} | {times['p90']:.3f} | "
                         f"{times['min']:.3f}—{times['max']:.3f} |")
    lines.extend(["", "固定 5 秒目标并不固定训练步数。实际完成步数和各段计时均保留，不能把训练不充分、系统负载变化或额外诊断耗时归因为某个网络模块的质量。此前固定 1600 步补充实验也不能按这些记录的步数直接折算为固定运行速度。", ""])


def append_instance_table(lines, title, summary):
    lines.extend([f"### {title}逐实例结果", "", "| 实例 | 定义 | 低度贪心 | 随机单次均值 | 随机重启均值 | 神经网络三个种子 | 神经均值 | SCIP 下界／上界 | SCIP 状态 |",
                  "|---|---|---:|---:|---:|---|---:|---|---|"])
    for row in summary["instance_results"]:
        means, scip = row["method_mean_sizes"], row["scip"]
        def size(method):
            return f"{means[method]:.2f}" if method in means else "—"
        bound = "—" if scip is None else f"{scip['size']}／{scip['upper_bound']:.3f}"
        status = "—" if scip is None else scip["status"]
        lines.append(f"| {row['name']} | {LABELS[row['definition']]} | {size('degree_greedy')} | "
                     f"{size('random_score_same_decoder')} | {size('random_restart_same_decoder')} | "
                     f"{' / '.join(map(str, row['neural_sizes']))} | {means[NEURAL]:.2f} | {bound} | {status} |")
    lines.extend(["", "SCIP 下界是其已找到的可行解大小，上界来自同一模型的对偶界；timelimit 表示截止时未证明最优，不能把该下界称为最大独立集大小。每条原始记录还保留 mip_gap、构建时间、求解时间和热启动大小。", ""])


def make_report(plan, selection, validation, test, real=None):
    title = "# 受控消融、锁定测试与公开真实数据实验" if real is not None else "# 受控消融与锁定测试实验"
    summaries = [validation, test] + ([real] if real is not None else [])
    count_line = (f"本报告完整核验 {sum(s['cases'] for s in summaries)} 个定义／实例场景、"
                  f"{sum(s['records'] for s in summaries)} 条最终解记录和 "
                  f"{sum(s['training_runs'] for s in summaries)} 次训练。")
    test_groups = {row["definition"]: row for row in test["aggregate"]}
    headline = "锁定测试集上，相对低度贪心的平均实例差异为" + "、".join(
        f"{LABELS[mode]} {percent(test_groups[mode]['comparisons']['degree_greedy']['mean_paired_percent'])}"
        for mode in ("weak", "strong")) + "；相对随机重启分别为" + "、".join(
        f"{LABELS[mode]} {percent(test_groups[mode]['comparisons']['random_restart_same_decoder']['mean_paired_percent'])}"
        for mode in ("weak", "strong")) + "。这些均为描述性结果，各方法实际总时间不相等，不能据此证明严格等时间优势。"
    lines = [title, "", "日期：2026-10-18。", "",
             count_line, "", headline, "",
             "这一轮将此前全部合成与公开配置补充实验视为开发数据，使用新的验证图选择配置，再按锁定配置评价未参与调参的测试图。所有成功和不理想结果一并保留；不进行显著性检验。这是 BIPNN 公开架构的 MIS 适配与受控评价，不是正式论文 Max-Cut 性能表的完整复现。", "",
             "## 预先固定的协议", "",
             "- 验证集：200 节点，k∈{3,5}，m/n∈{2,4}，实例种子 20261020，两种定义，每配置两个训练种子。对比宽度 256／1024 与 CRA 开／关的四种组合，其他设置固定。",
             "- 测试集：200／500 节点，k∈{3,5}，m/n∈{2,4}，实例种子 20261021、20261022，两种定义，每张图三个训练种子。测试结果不参与配置选择。m/n 是超边条数与节点数之比，不是平均关联度；k 均匀超图的平均关联度为 km/n。",
             "- 各定义分别按验证集“种子均值／低度贪心”的实例平均值选取配置；1e-12 内并列优先窄网络，再优先 CRA。",
             "- 网络仍直接使用未修改的作者公开 src.models 和 src.train 封装，固定的均匀随机节点特征、两层 HGNN+、BatchNorm、Adam 学习率 1e-4、权重衰减 5e-8、罚项系数 2。宽度 w 同时决定输入维度及 w、w/2 两个隐层，宽度消融包含这整个配置变化。公开 LINEAR 示例的 drop_rate=0.5 没有传递到其真实 dropout 参数，因此本轮如实使用有效 dropout=0。",
             "- 消息传播超边来自目标单项式：弱分支为原超边加各节点奖励的单点项；强分支为唯一冲突点对加奖励单点项。因此强分支实际在二阶冲突结构上使用 HGNN+ 运算，不是保留高阶约束必要性的证据。",
             "- CRA 开启时系数为 -1+1.6×(已用构建和训练秒数／目标秒数)，关闭时为 0。时间进度退火和索引式多项式计算是本项目的适配，不能称为作者原始固定 1600 步实验。输出沿用公开代码最后一次参数更新前的训练态前向。",
             "- 主目标预算为 5 秒。神经网络在完整训练步之间检查停止，末次解码额外计入实际总时间；随机重启在完整试次后检查；SCIP 从 5 秒中扣除 Python 模型和贪心初解构建时间，求解器最少 1 毫秒。它们是接近的时间目标，不是严格相同截止时间。", "",
             "- 低度贪心按原输入超图的节点关联次数升序，度相同按节点编号升序，依次加入仍可行的节点。随机对照产生 [0,1) 均匀分数，使用与网络相同的阈值、移除冲突点和扩充解码器；重启对照保留目标时间内完整试次中最大的可行解。SCIP 求解直接二进制线性约束，并显式使用该贪心可行初解。", "",
             "锁定选择：" + "；".join(f"{LABELS[mode]} `{config}`" for mode, config in selection["chosen"].items()) + "。", "",
             "源码追溯：验证 manifest 的 git_commit_at_start 为启动时已有的 `f4b3210`；当时新增实验源码尚未提交，但已由 manifest 的 source_sha256 逐文件绑定。相同训练源码随后保存于 `e3df0f24633bdda9138ac88393e1acffa022d71b`。应使用这一源码提交并核对哈希，不能仅检出 f4b3210 就声称可以复现本轮。后续提交还会收录报告、结果与额外后处理基准，训练关键文件保持本轮绑定版本，作者原 src 未改。", "",
             f"验证 manifest binding：`{validation['manifest_binding']}`。测试 manifest binding：`{test['manifest_binding']}`。", "",
             "汇总前独立重算 manifest binding、每张合成图、配置／种子／方法的完整集合，重新核验原始概率阈值、修复中间解、扩充解、初始解与每个保存的训练检查点。记录不完整时脚本报错，不输出完整结论。", ""]
    append_results(lines, "验证集：四组消融均完整保留", validation)
    append_results(lines, "锁定测试集：不再根据测试结果选参", test)
    append_instance_table(lines, "锁定测试集", test)
    if real is not None:
        lines.extend(["## 公开真实数据的身份与适用范围", "",
                      "使用作者公开仓库固定版本的 Cora、High、Primary 三个文件，分别有 1330／1413、327／7818、242／12704 个节点／超边。Cora 文件边数与正式论文表 4 的 1503 不一致，必须标为公开仓库变体；High／Primary 的统计一致也不能证明逐边同一实例。原始采集与全部处理链尚未逐边确认。来源、哈希、header 和标签映射见 [data-provenance.md](data-provenance.md)。", "",
                      "High 的全部高阶弱约束、Primary 4956 条高阶边中的 4951 条已由文件中的二节点子边蕴含。因此它们用于真实实例求解比较，不能单独证明高阶可行性建模必需。导入保留全部合法超边，因为删除冗余约束还会改变连续损失与消息传播结构。Cora 和纯 3／5 阶合成图提供不同结构的检验。", "",
                      f"真实数据 manifest binding：`{real['manifest_binding']}`。所有真实实例沿用验证集已锁定配置，不用其结果调参。", ""])
        append_results(lines, "三个公开仓库真实数据版本", real)
        append_instance_table(lines, "真实数据", real)
        lines.extend(["### 真实实例的实际主解耗时", "",
                      "| 实例 | 定义 | 低度贪心秒 | 随机单次秒 | 随机重启秒 | 神经网络秒 | SCIP 秒 |",
                      "|---|---|---:|---|---|---|---:|"])
        for row in real["instance_results"]:
            times = row["method_actual_seconds"]
            def elapsed(method):
                data = times[method]
                return f"{data['mean']:.3f}（{data['min']:.3f}—{data['max']:.3f}）"
            lines.append(f"| {row['name']} | {LABELS[row['definition']]} | {times['degree_greedy']['mean']:.3f} | "
                         f"{elapsed('random_score_same_decoder')} | {elapsed('random_restart_same_decoder')} | "
                         f"{elapsed(NEURAL)} | {times['scip_milp_greedy_start']['mean']:.3f} |")
        primary = next(row for row in real["instance_results"] if row["name"] == "primary-upstream" and row["definition"] == "strong")
        random_times = primary["method_actual_seconds"]["random_restart_same_decoder"]
        neural_times = primary["method_actual_seconds"][NEURAL]
        lines.extend(["", "随机和神经网络栏为三个种子平均值（最小—最大）。旧 CPU 解码器在稠密真实图上的耗时不可忽略："
                      f"Primary 强定义的随机重启实际耗时为 {random_times['min']:.3f}—{random_times['max']:.3f} 秒，"
                      f"网络主运行实际耗时为 {neural_times['min']:.3f}—{neural_times['max']:.3f} 秒。"
                      "因此本批真实数据各方法实际时间明显不相等，不能将任何解大小优势表述为同为严格 5 秒的公平优势。额外初始与轨迹诊断时间还需另计。", "",
                      "真实数据结论只能描述上述具体实例、已锁定配置与已保存的实际耗时。SCIP 截止时的可行下界可能被其他方法超过；这种情况不代表超过了已证明的最优解。未来若使用快速解码器，应统一替换网络和随机重启的同一后处理，再按新协议重新测时与评价，不与本轮旧解码器结果混合。", ""])
    lines.extend(["## 解释边界与下一步", "",
                  "这批实验只评价 5 秒初始预算和已列出的节点规模、密度、边阶及来源。它不能证明任意预算和更大规模都有效，也不能从全部因素同时改变的旧补充实验中单独归因网络宽度或退火。验证集四组比较可以描述在这一共同训练协议下的配置差异，时间预算会使不同宽度完成的更新步数不同。同一生成种子下的不同密度实例会共享部分超边，逐实例平均是对这些指定条件的描述，不将它们解释为相互独立的统计样本。", "",
                  "强独立集在每条超边内最多选择一个点，可等价展开为唯一冲突点对；弱独立集仅禁止整条超边全部入选，保留高阶目标。两种定义分别报告，不把弱、强解大小互相比较。神经网络与随机分数解码器，以及低度贪心，输出可行极大独立集；SCIP 截止时的初解只保证可行。只有 SCIP 证明 optimal 时才已知该场景的最大大小。", "",
                  "后续若扩展为 1／10／30 秒曲线或更大规模，应先另行固定协议，再运行并保留全部结果。当前初始 5 秒结果不代替预算曲线和规模扩展检验。", "",
                  "## 复现命令", "", "在仓库根目录使用已配置的项目环境。新运行请使用新的结果目录，避免与此次冻结记录混合。", "", "```powershell",
                  ".venv/Scripts/python.exe scripts/run_controlled.py validation --output results/controlled-validation-rerun",
                  ".venv/Scripts/python.exe scripts/run_controlled.py select --output results/controlled-validation-rerun --selection results/controlled-validation-rerun/selection.json",
                  ".venv/Scripts/python.exe scripts/check_controlled_protocol.py results/controlled-validation-rerun --selection results/controlled-validation-rerun/selection.json",
                  ".venv/Scripts/python.exe scripts/run_controlled.py test --output results/controlled-test-rerun --selection results/controlled-validation-rerun/selection.json",
                  ".venv/Scripts/python.exe scripts/summarize_controlled.py --validation results/controlled-validation-rerun --test results/controlled-test-rerun --report docs/controlled-rerun.md",
                  "```", "", "同一目录恢复运行时协议与已绑定源文件哈希必须一致；改变配置或源代码应另用新目录。固定种子控制输入和初始化，时间预算下实际完成的训练步数仍会受系统负载影响，不能保证逐位相同解。环境安装说明见 [environment.md](environment.md)，公开数据来源与转换审计见 [data-provenance.md](data-provenance.md)。", ""])
    if real is not None:
        lines.extend(["加入真实数据后重新汇总：", "", "```powershell",
                      ".venv/Scripts/python.exe scripts/run_real.py --output results/real-rerun --validation results/controlled-validation-rerun --selection results/controlled-validation-rerun/selection.json",
                      ".venv/Scripts/python.exe scripts/summarize_controlled.py --validation results/controlled-validation-rerun --test results/controlled-test-rerun --real results/real-rerun --report docs/controlled-with-real-rerun.md",
                      "```", ""])
    lines.extend(["完整阶段的一键入口为 `scripts/run_controlled_all.ps1`，已做 PowerShell 语法检查。本轮实际按相同阶段手动顺序执行；尚未从空环境、新目录完整执行该一键脚本，不能把语法检查写成端到端复现验证。额外的 fast_decode／后处理 kernel 基准未用于本轮训练或替换本轮生产解码器，其性能结果应单独报告。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--real", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.validation / "manifest.json")["protocol"]["plan"]
    validation = audit_synthetic(args.validation, "validation", plan)
    selection = select_from_validation(validation, plan, args.validation / "selection.json")
    test = audit_synthetic(args.test, "test", plan, selection)
    require(validation["manifest"]["environment"] == test["manifest"]["environment"],
            "Validation and test environment configurations differ")
    shared = set(validation["manifest"]["source_sha256"]) & set(test["manifest"]["source_sha256"])
    require(all(validation["manifest"]["source_sha256"][name] == test["manifest"]["source_sha256"][name]
                for name in shared), "Validation and test used different shared source files")
    require(set(plan["validation"]["instance_seeds"]).isdisjoint(plan["test"]["instance_seeds"]),
            "Validation and test instance seeds overlap")
    real = audit_real(args.real, plan, selection) if args.real is not None else None
    if real is not None:
        require(validation["manifest"]["environment"] == real["manifest"]["environment"],
                "Validation and real environment configurations differ")
        common = set(validation["manifest"]["source_sha256"]) & set(real["manifest"]["source_sha256"])
        require(all(validation["manifest"]["source_sha256"][name] == real["manifest"]["source_sha256"][name]
                    for name in common), "Validation and real evaluation used different shared sources")
    val_summary, test_summary = suite_summary(validation), suite_summary(test)
    real_summary = suite_summary(real) if real is not None else None
    # Write outputs only after every required input has passed the audit.
    report = make_report(plan, selection, val_summary, test_summary, real_summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf8")
    write_json(args.validation / "summary.json", val_summary)
    write_json(args.test / "summary.json", test_summary)
    if real_summary is not None:
        write_json(args.real / "summary.json", real_summary)
    print(json.dumps({"validation_cases": val_summary["cases"], "test_cases": test_summary["cases"],
                      "training_runs": val_summary["training_runs"] + test_summary["training_runs"],
                      "records": val_summary["records"] + test_summary["records"],
                      "real_cases": real_summary["cases"] if real_summary else None,
                      "real_training_runs": real_summary["training_runs"] if real_summary else None,
                      "total_training_runs": val_summary["training_runs"] + test_summary["training_runs"] +
                                             (real_summary["training_runs"] if real_summary else 0),
                      "total_records": val_summary["records"] + test_summary["records"] +
                                       (real_summary["records"] if real_summary else 0),
                      "chosen": selection["chosen"], "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"Audit refused; no complete summary was produced: {error}")
