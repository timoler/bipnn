"""Report the public-code preset diagnostic separately from the timed suite."""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mis.verify_report import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("preset", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.preset/"cases").glob("*.json"))
    if len(files) != 6:
        raise ValueError(f"Expected all six diagnostic cases, found {len(files)}")
    rows = []
    for p in files:
        verify(p)
        report = json.loads(p.read_text(encoding="utf8"))
        records = report["records"]
        if sorted(r["seed"] for r in records) != [0,1,2]:
            raise ValueError("Missing diagnostic replicate")
        if any(r["settings"]["epochs"] != 1600 for r in records):
            raise ValueError("A short smoke run was mixed into full diagnostic")
        baseline = json.loads((args.suite/"cases"/p.name).read_text(encoding="utf8"))
        if baseline["instance"] != report["instance"]:
            raise ValueError("The diagnostic and baseline instances differ")
        degree = next(r["size"] for r in baseline["records"] if r["method"] == "degree_greedy")
        small = st.mean(r["size"] for r in baseline["records"] if r["method"] == "hgnnp_pubo_repair")
        public = st.mean(r["size"] for r in records)
        proven = [r["size"] for r in baseline["records"] if r.get("optimality_proven")]
        if proven and any(r["size"] > proven[0] for r in records):
            raise ValueError("Diagnostic contradicts a proven upper bound")
        rows.append({"name": report["instance"]["name"], "n": report["instance"]["n"],
                     "definition": report["definition"], "greedy_size": degree,
                     "small_preset_mean_size": small, "public_preset_mean_size": public,
                     "public_preset_sizes": [r["size"] for r in records],
                     "public_preset_raw_sizes": [r["raw_size"] for r in records],
                     "public_preset_raw_feasible_rate": st.mean(int(r["raw_feasible"]) for r in records),
                     "public_preset_mean_seconds": st.mean(r["total_seconds"] for r in records),
                     "public_vs_greedy_percent": 100*(public/degree-1),
                     "proven_optimum": proven[0] if proven else None})
    result = {"cases": rows, "training_runs": 18,
              "limits": "fixed subset, altered combined configuration and longer budget; not equal-time evidence or complete paper reproduction"}
    (args.preset/"summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    lines = ["# 作者公开配置的补充实验", "", "日期：2026-10-18。共 6 个场景，每个三个训练种子，合计 18 次完整训练。", "",
             "直接调用作者公开仓库未修改的网络封装、训练步和 PUBO 损失，使用公开 Notebook 中 Max-Cut 示例的配置：1024 维均匀随机输入、1024/512 两个隐层、学习率 1e-4、Adam 权重衰减 5e-8、1600 步、退火系数 -1+epoch/1000。网络输出沿用原训练函数最后一次更新前的训练前向结果。", "",
             "这里将目标函数适配为明确的强／弱超图独立集，并使用同一可行性后处理；因此属于公开配置的 MIS 适配诊断，不是原论文完整实验复现。", "",
             "固定选择每个规模的一张 k=3、m=2n、生成种子为 20261018 的超图，覆盖 50、200、500 节点和两种定义。选择规则不是在所有结果中挑最好的实例，但该子集也不能替代完整泛化评估。", "",
             "| 节点数 | 定义 | 低度贪心 | 小模型均值 | 公开配置三个种子的解大小 | 公开配置均值 | 原始解可行率 | 平均总耗时（秒） |",
             "|---:|---|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        label = "弱" if r["definition"] == "weak" else "强"
        lines.append(f"| {r['n']} | {label} | {r['greedy_size']} | {r['small_preset_mean_size']:.2f} | {' / '.join(map(str,r['public_preset_sizes']))} | {r['public_preset_mean_size']:.2f} | {r['public_preset_raw_feasible_rate']:.1%} | {r['public_preset_mean_seconds']:.2f} |")
    lines.extend(["", "## 解释范围", "",
                  "小模型使用约 2 秒预算，公开配置使用固定 1600 步，不能把表中差异解释为相同时间预算下的优势。网络规模、BatchNorm、输入分布、学习率、退火曲线和训练时长同时改变，不能将收益归因于单个模块。", "",
                  "后处理后的具体解、原始阈值解、概率及训练轨迹均保存，并独立检查可行性。该诊断用于选择后续应继续核验的配置，不构成最终论文性能结论。", "",
                  "公开配置来源：[作者 Notebook](https://github.com/Classpi/BIPNN-Learning-to-Solve-Binary-Integer-Programming-via-Hypergraph-Neural-Networks/blob/708c10a95a50fb2b06dd35bf1604e65a6ec5ebe5/example/desc.ipynb)。", ""])
    args.report.write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
