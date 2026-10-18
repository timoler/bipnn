"""Run predeclared validation, select configurations, then evaluate new graphs."""
import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mis.benchmark import generate_instance, random_decoder, write_json
from mis.controlled import CONFIGS, bind_manifest, fingerprint, solve_controlled
from mis.problem import HypergraphProblem, degree_greedy
from mis.reference import solve_scip
from mis.verify_report import verify


def cases(settings, definitions):
    for n, k, density, seed in itertools.product(settings["sizes"], settings["orders"],
                                                 settings["densities_m_over_n"], settings["instance_seeds"]):
        name = f"n{n}_k{k}_d{density}_s{seed}"
        instance = {"name": name, **generate_instance(n, k, density, seed)}
        for mode in definitions:
            yield name, mode, instance


def identity(record):
    return (record["method"], record.get("config"), record.get("seed"))


def expected(phase, settings, mode, selection):
    result = [("degree_greedy", None, None)]
    configs = settings["configs"] if phase == "validation" else [selection["chosen"][mode]]
    if phase == "test":
        result.append(("scip_milp_greedy_start", None, None))
        for seed in settings["model_seeds"]:
            result.extend([("random_score_same_decoder", None, seed),
                           ("random_restart_same_decoder", None, seed)])
    for config in configs:
        for seed in settings["model_seeds"]:
            result.append(("public_architecture_indexed_budget", config, seed))
    return result


def run(args, plan):
    settings = plan[args.phase]
    selection = None
    if args.phase == "test":
        selection = json.loads(args.selection.read_text(encoding="utf8"))
        if selection["plan_sha256"] != fingerprint(plan):
            raise ValueError("Selection belongs to another protocol")
    protocol = {"plan": plan, "phase": args.phase, "selection": selection}
    torch.set_num_threads(1)
    # Same warmup for both phases, excluded from trial wall times.
    warm = HypergraphProblem(6, ((0, 1, 2), (3, 4)))
    solve_controlled(warm, "weak", "w256_cra", 0, epochs=2)
    solve_scip(warm, "weak", 0.1, total_budget=True)
    manifest = bind_manifest(args.output, protocol)
    for name, mode, instance in cases(settings, plan["definitions"]):
        path = args.output / "cases" / f"{name}_{mode}.json"
        signature = fingerprint(instance)
        if path.exists():
            report = json.loads(path.read_text(encoding="utf8"))
            if report["manifest_binding"] != manifest["binding"] or fingerprint(report["instance"]) != signature:
                raise ValueError("Stored case differs from bound protocol/instance")
            verify(path)
        else:
            report = {"definition": mode, "instance": instance, "records": [],
                      "manifest_binding": manifest["binding"], "purpose": args.phase,
                      "environment": manifest["environment"]}
        problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
        required = expected(args.phase, settings, mode, selection)
        existing = [identity(r) for r in report["records"]]
        if len(set(existing)) != len(existing) or any(item not in required for item in existing):
            raise ValueError("Unexpected or duplicated records in saved case")
        for method, config, seed in required:
            if (method, config, seed) in existing:
                continue
            if method == "degree_greedy":
                start = time.perf_counter()
                chosen = degree_greedy(problem, mode)
                record = {"method": method, "selected": chosen, "size": len(chosen),
                          "feasible": True, "total_seconds": time.perf_counter() - start}
            elif method == "scip_milp_greedy_start":
                record = solve_scip(problem, mode, plan["budget_seconds"], total_budget=True)
            elif method.startswith("random_"):
                budget = plan["budget_seconds"] if "restart" in method else None
                record = random_decoder(problem, mode, seed, budget)
            else:
                record = solve_controlled(problem, mode, config, seed, budget=plan["budget_seconds"])
            report["records"].append(record)
            write_json(path, report)
            verify(path)
            print(f"{args.phase.upper()} {name} {mode} {method} {config} seed={seed} "
                  f"size={record['size']} seconds={record['total_seconds']:.3f} "
                  f"epochs={record.get('epochs_completed', '-')}", flush=True)


def select(args, plan):
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf8"))
    if manifest["protocol"]["plan"] != plan or manifest["protocol"]["phase"] != "validation":
        raise ValueError("Validation manifest differs from protocol")
    ratios = {mode: {config: [] for config in plan["validation"]["configs"]} for mode in plan["definitions"]}
    for name, mode, instance in cases(plan["validation"], plan["definitions"]):
        path = args.output / "cases" / f"{name}_{mode}.json"
        verify(path)
        report = json.loads(path.read_text(encoding="utf8"))
        if report["manifest_binding"] != manifest["binding"] or fingerprint(report["instance"]) != fingerprint(instance):
            raise ValueError("Validation source or instance mismatch")
        if sorted(identity(r) for r in report["records"]) != sorted(expected("validation", plan["validation"], mode, None)):
            raise ValueError("Validation is incomplete")
        degree = next(r["size"] for r in report["records"] if r["method"] == "degree_greedy")
        for config in ratios[mode]:
            values = [r["size"] for r in report["records"] if r.get("config") == config]
            ratios[mode][config].append(statistics.mean(values) / degree)
    scores = {mode: {config: statistics.mean(values) for config, values in configs.items()} for mode, configs in ratios.items()}
    chosen = {}
    for mode, values in scores.items():
        top = max(values.values())
        tied = [c for c in values if top - values[c] <= 1e-12]
        chosen[mode] = min(tied, key=lambda c: (CONFIGS[c]["width"], CONFIGS[c]["annealing"] != "cra"))
    selection = {"plan_sha256": fingerprint(plan), "validation_binding": manifest["binding"],
                 "criterion": plan["selection"], "validation_scores": scores, "chosen": chosen}
    path = args.selection
    if path.exists() and json.loads(path.read_text(encoding="utf8")) != selection:
        raise ValueError("Refusing to replace an already locked selection")
    write_json(path, selection)
    print(json.dumps(selection, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validation", "select", "test"])
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/controlled-20261018.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=ROOT / "results/controlled-validation-20261018/selection.json")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    if args.phase == "select":
        select(args, plan)
    else:
        run(args, plan)


if __name__ == "__main__":
    main()
