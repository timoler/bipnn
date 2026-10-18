"""Audit every saved solution and summarize matched instances, not pooled seeds."""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mis.problem import HypergraphProblem
from mis.verify_report import verify

LABELS = {
    "degree_greedy": "低度优先贪心",
    "highs_milp": "HiGHS（限时）",
    "scip_milp_greedy_start": "SCIP（贪心初始解）",
    "random_score_same_decoder": "随机分数＋同一后处理",
    "random_restart_same_decoder": "限时随机重启＋同一后处理",
    "hgnnp_pubo_repair": "HGNN+＋PUBO＋后处理",
}


def mean(values):
    return st.mean(values)


def summarize(suite, require_complete=True):
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf8"))
    settings = manifest["settings"]
    expected = len(settings["sizes"]) * len(settings["orders"]) * len(settings["densities"]) * len(settings["instance_seeds"]) * len(settings["definitions"])
    files = sorted((suite / "cases").glob("*.json"))
    if require_complete and len(files) != expected:
        raise ValueError(f"Suite incomplete: {len(files)}/{expected} cases")
    cases = []
    record_count = 0
    for path in files:
        verify(path)
        report = json.loads(path.read_text(encoding="utf8"))
        if report["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Mixed source versions in results")
        instance = report["instance"]
        instance_path = suite / "instances" / (instance["name"] + ".json")
        import hashlib
        if hashlib.sha256(instance_path.read_bytes()).hexdigest() != instance["sha256"]:
            raise ValueError("Instance hash does not match saved report")
        problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
        by_method = defaultdict(list)
        proven = []
        best = max(r["size"] for r in report["records"])
        record_count += len(report["records"])
        for r in report["records"]:
            if not problem.feasible(r["selected"], report["definition"]):
                raise ValueError("Infeasible output")
            if r["method"] not in ("highs_milp", "scip_milp_greedy_start"):
                chosen = set(r["selected"])
                blocked = set()
                for edge in problem.edges:
                    missing = set(edge) - chosen
                    if report["definition"] == "weak" and len(missing) == 1:
                        blocked.update(missing)
                    if report["definition"] == "strong" and len(missing) < len(edge):
                        blocked.update(missing)
                if set(range(problem.n)) - chosen - blocked:
                    raise ValueError("Postprocessed/greedy solution is feasible but not maximal")
            if r.get("upper_bound") is not None and r["upper_bound"] + 1e-5 < best:
                raise ValueError("Claimed bound contradicts a saved feasible solution")
            if r.get("optimality_proven"):
                proven.append(r["size"])
            by_method[r["method"]].append(r)
        if proven and (len(set(proven)) != 1 or best > proven[0]):
            raise ValueError("Inconsistent proof of optimality")
        if set(by_method) != set(LABELS):
            raise ValueError("Missing method in case")
        methods = {}
        for method, records in by_method.items():
            expected_repeats = len(settings["model_seeds"]) if method in ("hgnnp_pubo_repair", "random_score_same_decoder", "random_restart_same_decoder") else 1
            if len(records) != expected_repeats:
                raise ValueError("Missing replicate")
            if expected_repeats > 1 and sorted(r["seed"] for r in records) != sorted(settings["model_seeds"]):
                raise ValueError("Repeated or missing random seed")
            methods[method] = {
                "mean_size": mean([r["size"] for r in records]),
                "sizes": [r["size"] for r in records],
                "mean_seconds": mean([r["total_seconds"] for r in records]),
                "sizes_over_best_found": mean([r["size"] / best for r in records]),
                "optimality_proven": any(r.get("optimality_proven", False) for r in records),
            }
            if "raw_size" in records[0]:
                methods[method].update({
                    "mean_raw_size": mean([r["raw_size"] for r in records]),
                    "raw_feasible_rate": mean([int(r["raw_feasible"]) for r in records]),
                    "raw_empty_rate": mean([int(r["raw_size"] == 0) for r in records]),
                    "mean_postprocessing_gain": mean([r["size"]-r["raw_size"] for r in records]),
                })
        cases.append({"name": instance["name"], "definition": report["definition"],
                      "n": instance["n"], "generator": instance["generator"],
                      "best_found": best, "proven_optimum": proven[0] if proven else None,
                      "methods": methods})
    aggregate = {}
    for mode in settings["definitions"]:
        subset = [c for c in cases if c["definition"] == mode]
        if not subset:
            continue
        nn = "hgnnp_pubo_repair"
        greedy = "degree_greedy"
        random_single = "random_score_same_decoder"
        delta = [100*(c["methods"][nn]["mean_size"] / c["methods"][greedy]["mean_size"]-1) for c in subset]
        training_delta = [100*(c["methods"][nn]["mean_size"] / c["methods"][random_single]["mean_size"]-1) for c in subset]
        aggregate[mode] = {
            "cases": len(subset), "proven_cases": sum(c["proven_optimum"] is not None for c in subset),
            "nn_vs_degree_mean_percent": mean(delta),
            "nn_vs_random_single_mean_percent": mean(training_delta),
            "nn_degree_wins_ties_losses": [sum(x > 1e-9 for x in delta), sum(abs(x) <= 1e-9 for x in delta), sum(x < -1e-9 for x in delta)],
            "nn_raw_empty_rate": mean([c["methods"][nn]["raw_empty_rate"] for c in subset]),
            "nn_raw_feasible_rate": mean([c["methods"][nn]["raw_feasible_rate"] for c in subset]),
            "methods": {m: {
                "mean_ratio_to_best_found": mean([c["methods"][m]["sizes_over_best_found"] for c in subset]),
                "mean_seconds": mean([c["methods"][m]["mean_seconds"] for c in subset]),
            } for m in LABELS},
        }
    return {"expected_cases": expected, "completed_cases": len(cases), "records_verified": record_count,
            "config": settings, "aggregate": aggregate, "cases": cases}


def report_markdown(summary):
    mode_label = {"weak": "弱独立集", "strong": "强独立集"}
    lines = ["# 第一轮合成超图实验", "", "实验日期：2026-10-18。所有记录包含具体解并经过独立可行性检查。", "",
             f"已完成 {summary['completed_cases']}/{summary['expected_cases']} 个实例与定义组合，核验 {summary['records_verified']} 条方法运行记录。", "",
             "## 实验设置", "",
             "24 张固定合成超图：节点数 50、200、500；超边大小 3、5；超边数为节点数的 2 倍或 4 倍；每种配置两个生成种子。每张图分别研究强、弱两种定义。每个随机方法使用三个种子，神经网络不跨实例共享训练。", "",
             "神经方法与随机重启的搜索目标为每次约 2 秒。神经方法预算包含模型与结构构建和训练，最后预测／后处理另计；随机重启在完整一次解码后检查时限。HiGHS 与 SCIP 的限制为求解器内部 2 秒，其建模及初始解开销另计。以保存的实际总耗时为准，不能称为严格等时实验。", "",
             "SCIP 显式使用低度贪心初始解、单线程。HiGHS 没有注入该初始解。单次随机分数对照与神经方法采用完全相同的解码器，用来检查训练是否提供额外排序价值；该对照不使用满 2 秒。", "",
             "## 按定义汇总", ""]
    for mode, a in summary["aggregate"].items():
        lines.extend([f"### {mode_label[mode]}", "",
                      f"共 {a['cases']} 个匹配实例，其中 {a['proven_cases']} 个有求解器证明最优值。", "",
                      f"神经方法相对低度贪心的实例平均解大小差异为 {a['nn_vs_degree_mean_percent']:+.2f}%；胜／平／负为 {' / '.join(map(str,a['nn_degree_wins_ties_losses']))}。", "",
                      f"相对单次随机分数＋同一后处理，平均差异为 {a['nn_vs_random_single_mean_percent']:+.2f}%。网络原始阈值解可行率为 {a['nn_raw_feasible_rate']:.1%}，空集比例为 {a['nn_raw_empty_rate']:.1%}。", "",
                      "| 方法 | 相对本批最好可行解的平均比例 | 平均总耗时（秒） |", "|---|---:|---:|"])
        for method, stats in a["methods"].items():
            lines.append(f"| {LABELS[method]} | {stats['mean_ratio_to_best_found']:.3f} | {stats['mean_seconds']:.3f} |")
        lines.append("")
    lines.extend(["## 解释范围", "",
                  "先在每个实例内对三个随机种子取均值，再对匹配实例汇总；不把重复训练种子当成独立超图。不同稠密度在相同生成种子下可能共享部分超边，汇总只作描述性比较，不报告显著性检验。", "",
                  "表中的最好可行解来自本批所有方法；没有最优性证明时，它不是数学最优解。原始解可行率必须结合空集比例解读，空集总是可行，但没有求解质量。", "",
                  "本轮固定使用 32 维特征、两层 HGNN+、学习率 0.01、惩罚系数 2 和最终二值化权重 0.1；不是经过系统调优的最佳 BIPNN 配置，也不代表原论文的完整复现。全部配置与失败表现均保留。", "",
                  "PySCIPOpt 对当前链接的 SCIP 版本给出推荐版本不匹配提醒；小实例与枚举对照通过，但后续正式实验应固定匹配版本。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = summarize(args.suite, require_complete=not args.partial)
    (args.suite / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_markdown(result), encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("cases", "config")}, ensure_ascii=False, indent=2))
