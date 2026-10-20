"""Plot the complete, audited thesis controls and new-graph replication.

No experiments, GPU calls, selection of a best seed, or changes to raw data.
Graph seeds are reused across size/structure conditions; ranges are conditional,
descriptive graph ranges, not globally IID uncertainty or confidence intervals.
Panel d uses the separately audited targets[5,15] child at 15 seconds to correct
the original sampling-calendar mismatch; the unmatched result stays diagnostic.
Usage: python scripts/plot_completion.py --qa-tools <QA tools directory>
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = ["S3W", "S5W", "CoraW", "CoraS", "HighW", "HighS", "PrimaryW", "PrimaryS"]
WEAK_CASES = ["S3W", "S5W", "CoraW", "HighW", "PrimaryW"]
METHODS = ["neural", "random", "scip", "degree"]
STYLE = {
    "neural": {"label": "BIPNN (adapted)", "color": "#286F8E", "marker": "o", "linestyle": "-"},
    "random": {"label": "Random restart", "color": "#8666A5", "marker": "D", "linestyle": "-."},
    "scip": {"label": "SCIP", "color": "#BB742D", "marker": "s", "linestyle": "--"},
    "degree": {"label": "Degree greedy", "color": "#777777", "marker": "^", "linestyle": ":"},
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def case_label(row):
    generator = row["generator"]
    if generator is not None:
        signature = (row["n"], generator["k"], generator["density_m_over_n"], generator["seed"], row["definition"])
        if signature == (500, 3, 2, 20261023, "weak"):
            return "S3W"
        if signature == (500, 5, 4, 20261023, "weak"):
            return "S5W"
        raise ValueError("Unexpected ablation graph; do not silently exclude it")
    name = row["name"].removesuffix("-upstream")
    names = {"cora": "Cora", "high": "High", "primary": "Primary"}
    if name not in names or row["definition"] not in ("weak", "strong"):
        raise ValueError("Unexpected real ablation identity")
    return names[name] + ("W" if row["definition"] == "weak" else "S")


def load_summary(path):
    raw = path.read_bytes()
    summary = json.loads(raw)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf8"))
    if summary.get("complete") is not True or summary["manifest_binding"] != manifest["binding"]:
        raise ValueError("Final plotting requires a complete summary with matching manifest")
    rows = summary["neural_rows"]
    if len(rows) != 303 or Counter(row["group"] for row in rows) != {"fixed": 93, "timed": 174, "replication": 36}:
        raise ValueError("Missing or additional neural primary summary rows")
    index = {}
    for row in rows:
        if not 0 <= row["size"] <= row["n"] or row["seed"] not in (0, 1, 2):
            raise ValueError("Invalid model seed or feasible solution size")
        if row["group"] == "replication":
            continue
        key = (row["group"], case_label(row), row["annealing"], float(row["penalty"]),
               row["seed"], row["horizon_seconds"], row["cutoff_seconds"])
        if key in index:
            raise ValueError("Duplicated ablation identity")
        if row["group"] == "fixed" and row["fixed_epochs"] != 1600:
            raise ValueError("Fixed-update count differs from the panel contract")
        index[key] = row
    matched = summary["schedule_match"]
    child_manifest = json.loads((ROOT / matched["directory"] / "manifest.json").read_text(encoding="utf8"))
    if (matched.get("complete") is not True or matched["manifest_binding"] != child_manifest["binding"]
            or matched["parent_manifest_binding"] != manifest["binding"]):
        raise ValueError("Sampling-corrected schedule child is incomplete or its source binding differs")
    child_rows = matched["neural_rows"]
    child_index = {}
    for row in child_rows:
        if (row["group"] != "schedule_match" or row["annealing"] != "cra" or row["penalty"] != 2
                or row["definition"] != "weak" or row["horizon_seconds"] != 15
                or row["targets_seconds"] != [5, 15] or row["sampling_calendar_verified"] is not True):
            raise ValueError("Child row does not certify the matched pre-15-second sampling calendar")
        key = ("timed", case_label(row), "cra", 2., row["seed"], 15., row["cutoff_seconds"])
        if key in child_index or not 0 <= row["size"] <= row["n"]:
            raise ValueError("Duplicated or invalid sampling-corrected child result")
        child_index[key] = row
    expected_child = {("timed", label, "cra", 2., seed, 15., cutoff)
                      for label in WEAK_CASES for seed in (0, 1, 2) for cutoff in (5., 15.)}
    if len(child_rows) != 30 or set(child_index) != expected_child:
        raise ValueError("Sampling-corrected child must retain all 15 trajectories and 30 cutoff records")
    return summary, manifest, index, child_index, hashlib.sha256(raw).hexdigest()


def row_for(index, group, label, annealing="cra", penalty=2., seed=0, horizon=None, cutoff=None):
    return index[(group, label, annealing, float(penalty), seed, horizon, cutoff)]


def paired(index, group, label, treatment, reference, reference_index=None):
    differences, evidence = [], []
    for seed in (0, 1, 2):
        a = row_for(index, group, label, seed=seed, **treatment)
        b = row_for(index if reference_index is None else reference_index, group, label, seed=seed, **reference)
        for field in ("initial_state_sha256", "feature_sha256"):
            digest = a[field]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or digest != b[field]:
                raise ValueError(f"Paired initialization/feature mismatch: {label}, seed {seed}, {field}")
        if a["n"] != b["n"] or a["definition"] != b["definition"]:
            raise ValueError("Pair combines different problem definitions or node counts")
        delta = 100 * (a["size"]-b["size"]) / a["n"]
        differences.append(delta)
        evidence.append({"seed": seed, "difference_pp": delta, "treatment": a, "reference": b})
    return {"case": label, "mean": float(st.mean(differences)), "minimum": min(differences),
            "maximum": max(differences), "paired_seed_differences_pp": differences, "evidence": evidence}


def error_points(axis, x, values, *, color, marker="o", label=None, linestyle="none", size=3.7):
    center = np.asarray([v["mean"] for v in values])
    low = np.asarray([v["minimum"] for v in values])
    high = np.asarray([v["maximum"] for v in values])
    if np.any(low > center) or np.any(high < center):
        raise ValueError("Center lies outside the retained observed range")
    axis.errorbar(x, center, yerr=np.vstack((center-low, high-center)), fmt=marker,
                   color=color, label=label, linestyle=linestyle, markersize=size,
                   linewidth=1., elinewidth=0.7, capsize=2., capthick=0.7)


def export(fig, stem, qa_tools, panel_ids):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=fig.axes, panel_ids=panel_ids, json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True)
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".svg")
    fig.savefig(str(stem) + ".png", dpi=400)
    plt.close(fig)
    run_audit(qa_tools, "audit_pdf_text.py", [str(stem) + ".pdf", "--min-pt", "5", "--json"],
              Path(str(stem) + ".font-audit.json"))
    run_audit(qa_tools, "audit_figure_collisions.py",
              [str(stem) + ".pdf", "--json-out", str(stem) + ".collision-audit.json",
               "--overlay-pdf", str(stem) + ".collision-audit.pdf"], Path(str(stem) + ".collision-audit.txt"))


def panel_label(axis, label, offset=8):
    axis.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-27, offset),
                  textcoords="offset points", fontsize=9, fontweight="bold",
                  ha="left", va="bottom", annotation_clip=False)


def ablation_figure(index, child_index, output, qa_tools):
    a = [paired(index, "fixed", label, {"annealing": "cra"}, {"annealing": "none"}) for label in CASES]
    b = [paired(index, "timed", label, {"annealing": "cra", "horizon": 30., "cutoff": 30.},
                {"annealing": "none", "horizon": 30., "cutoff": 30.}) for label in CASES]
    c = {str(p): [paired(index, "fixed", label, {"penalty": p}, {"penalty": 2.}) for label in WEAK_CASES]
         for p in (1.1, 2., 4., 8.)}
    d = {
        "5.0": [paired(index, "timed", label, {"horizon": 5., "cutoff": 5.},
                       {"horizon": 30., "cutoff": 5.}) for label in WEAK_CASES],
        "15.0": [paired(child_index, "timed", label, {"horizon": 15., "cutoff": 15.},
                        {"horizon": 30., "cutoff": 15.}, reference_index=index) for label in WEAK_CASES],
    }
    width_mm, height_mm = 180, 155
    fig, axes = plt.subplots(2, 2, figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.12, top=0.93, wspace=0.34, hspace=0.90)
    error_points(axes[0, 0], np.arange(8), a, color="#286F8E")
    error_points(axes[0, 1], np.arange(8), b, color="#286F8E")
    penalty_styles = [("1.1", "#286F8E", "o", -0.24), ("2.0", "#777777", "D", -0.08),
                      ("4.0", "#8666A5", "s", 0.08), ("8.0", "#BB742D", "^", 0.24)]
    for penalty, color, marker, offset in penalty_styles:
        error_points(axes[1, 0], np.arange(5)+offset, c[penalty], color=color,
                      marker=marker, label=f"λ = {float(penalty):g}", size=3.4)
    for horizon, color, marker, offset in (("5.0", "#286F8E", "o", -0.10), ("15.0", "#BB742D", "s", 0.10)):
        error_points(axes[1, 1], np.arange(5)+offset, d[horizon], color=color, marker=marker,
                      label=f"{float(horizon):g}-second horizon", size=3.5)
    all_values = [a, b, [v for values in c.values() for v in values], [v for values in d.values() for v in values]]
    titles = ["CRA − none | 1,600 updates", "CRA − none | 30-second cutoff",
              "Penalty sensitivity | 1,600 updates", "CRA horizon at matched cutoffs"]
    for i, axis in enumerate(axes.flatten()):
        labels = CASES if i < 2 else WEAK_CASES
        axis.set_xlim(-0.6, len(labels)-0.4)
        axis.set_xticks(range(len(labels)), labels)
        if i < 2:
            plt.setp(axis.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor", fontsize=6.5)
        axis.set_ylabel("Δ feasible size / n (pp)", labelpad=5)
        axis.set_title(titles[i], fontsize=7.5, pad=8 if i < 2 else 47)
        axis.axhline(0, color="#676767", linewidth=0.75, linestyle=(0, (3, 3)), zorder=0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E7E7E7", linewidth=0.5)
        lows = [0.] + [v["minimum"] for v in all_values[i]]
        highs = [0.] + [v["maximum"] for v in all_values[i]]
        pad = max(0.2, (max(highs)-min(lows))*0.15)
        axis.set_ylim(min(lows)-pad, max(highs)+pad)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        panel_label(axis, chr(ord("a")+i), 8)
    axes[1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.31), ncol=2,
                       frameon=False, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.5)
    axes[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.31), ncol=1,
                       frameon=False, handletextpad=0.5, borderaxespad=0.5)
    fig.text(0.54, 0.025, "Paired seed differences; min–max ranges are not confidence intervals.",
             fontsize=7, ha="center", va="bottom")
    export(fig, output / "ablation-controls", qa_tools, ["a", "b", "c", "d"])
    return {"a": a, "b": b, "c": c, "d": d}


def replication_figure(rows, output, qa_tools):
    expected = {(n, k, density, seed) for n in (1000, 2000) for k, density in ((3, 2), (5, 4))
                for seed in (20261027, 20261028, 20261029)}
    index = {}
    for row in rows:
        g = row["generator"]
        key = (row["n"], g["k"], g["density_m_over_n"], g["seed"])
        if row["definition"] != "weak" or key in index or row["model_seeds"] != [0, 1, 2]:
            raise ValueError("Unexpected or duplicated new-graph identity")
        for field in ("neural_sizes", "random_sizes"):
            if len(row[field]) != 3 or any(not 0 <= x <= row["n"] for x in row[field]):
                raise ValueError("Missing model seed in graph-level aggregation")
        index[key] = row
    if set(index) != expected or len(rows) != 12:
        raise ValueError("Require the 12 new graphs, three graph seeds per size/structure, without old graph mixing")
    width_mm, height_mm = 180, 88
    fig, axes = plt.subplots(1, 2, figsize=(width_mm / 25.4, height_mm / 25.4), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.23, top=0.805, wspace=0.28)
    panels, all_lows, all_highs = {}, [], []
    for i, (k, density) in enumerate(((3, 2), (5, 4))):
        axis, curves = axes[i], {}
        for method in METHODS:
            points = []
            for n in (1000, 2000):
                graph_values, graph_evidence = [], []
                for seed in (20261027, 20261028, 20261029):
                    row = index[(n, k, density, seed)]
                    size = (float(np.mean(row["neural_sizes"])) if method == "neural" else
                            float(np.mean(row["random_sizes"])) if method == "random" else
                            float(row["scip_size"]) if method == "scip" else float(row["degree_size"]))
                    value = size/n
                    if not 0 <= value <= 1:
                        raise ValueError("Graph quality outside [0,1]")
                    graph_values.append(value)
                    graph_evidence.append({"generator_seed": seed, "graph_mean_fraction": value, "source_row": row})
                points.append({"n": n, "mean": float(st.mean(graph_values)), "minimum": min(graph_values),
                               "maximum": max(graph_values), "graph_values": graph_values, "evidence": graph_evidence})
            style = STYLE[method]
            error_points(axis, [1000, 2000], points, color=style["color"], marker=style["marker"],
                          label=style["label"], linestyle=style["linestyle"])
            curves[method] = points
            all_lows.extend(v["minimum"] for v in points)
            all_highs.extend(v["maximum"] for v in points)
        axis.set_xticks([1000, 2000], ["1,000", "2,000"])
        axis.set_xlim(850, 2150)
        axis.set_xlabel("Nodes n", labelpad=5)
        if i == 0:
            axis.set_ylabel("Feasible set size / n", labelpad=5)
        axis.set_title(f"k = {k}, m/n = {density} | 30-second cutoff", fontsize=7.5, pad=8)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.2f}"))
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E7E7E7", linewidth=0.5)
        panel_label(axis, chr(ord("a")+i), 8)
        panels[chr(ord("a")+i)] = {"k": k, "density_m_over_n": density, "curves": curves}
    pad = max(0.015, (max(all_highs)-min(all_lows))*0.1)
    axes[0].set_ylim(max(0., min(all_lows)-pad), min(1., max(all_highs)+pad))
    handles = [Line2D([], [], label=STYLE[m]["label"], color=STYLE[m]["color"], marker=STYLE[m]["marker"],
                       linestyle=STYLE[m]["linestyle"], linewidth=1., markersize=3.5) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=4,
               frameon=False, handlelength=2., handletextpad=0.5, columnspacing=1.3)
    fig.text(0.54, 0.025, "Graph means and graph min–max ranges; three new graphs per point.",
             fontsize=7, ha="center", va="bottom")
    export(fig, output / "independent-graphs", qa_tools, ["a", "b"])
    return panels


def run_audit(tools, filename, arguments, output):
    result = subprocess.run([sys.executable, str(tools / filename), *map(str, arguments)],
                            capture_output=True, text=True, encoding="utf8", env={**os.environ, "PYTHONUTF8": "1"})
    output.write_text(result.stdout+result.stderr, encoding="utf8")
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"Figure audit failed: {filename}; see {output.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "results/completion-20261020/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/completion-20261020/figures")
    parser.add_argument("--qa-tools", type=Path, required=True)
    args = parser.parse_args()
    summary, manifest, index, child_index, source_hash = load_summary(args.summary)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.qa_tools))
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.65, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    ablation = ablation_figure(index, child_index, args.output, args.qa_tools)
    replication = replication_figure(summary["replication_rows"], args.output, args.qa_tools)
    write_json(args.output / "source-data.json", {"summary_source": "../summary.json", "summary_sha256": source_hash,
        "plotting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_binding": manifest["binding"], "excluded_summary_rows": 0,
        "complete_audited_summary": summary,
        "neural_rows": summary["neural_rows"], "fixed_rows": summary["fixed_rows"],
        "timed_rows": summary["timed_rows"], "replication_rows": summary["replication_rows"],
        "schedule_match": summary["schedule_match"],
        "schedule_panel_rule": "5s: original matched h5 versus h30; 15s: corrected child targets[5,15] versus h30. Original unmatched targets[15] is diagnostic only; child 5s records are also retained.",
        "ablation_panels": ablation, "replication_panels": replication,
        "ablation_range": "Minimum to maximum of three matched model-seed differences; percentage points",
        "replication_range": "Within each size/structure: minimum to maximum of three graph means after graph-internal model-seed averaging; not a confidence interval",
        "replication_dependence": "The three graph seeds are reused across size/structure conditions; do not treat all 12 graphs as unrelated IID samples"})
    write_csv(args.output / "source-main-neural.csv", summary["neural_rows"])
    write_csv(args.output / "source-schedule-match.csv", summary["schedule_match"]["neural_rows"])
    write_csv(args.output / "source-replication-graphs.csv", summary["replication_rows"])
    differences = []
    for panel, groups in ablation.items():
        variants = {"CRA_minus_none": groups} if panel in ("a", "b") else groups
        for variant, points in variants.items():
            for point in points:
                for evidence in point["evidence"]:
                    differences.append({"panel": panel, "variant": variant, "case": point["case"],
                        "seed": evidence["seed"], "difference_pp": evidence["difference_pp"],
                        "treatment_size": evidence["treatment"]["size"],
                        "reference_size": evidence["reference"]["size"], "n": evidence["treatment"]["n"],
                        "initial_state_sha256": evidence["treatment"]["initial_state_sha256"],
                        "feature_sha256": evidence["treatment"]["feature_sha256"]})
    write_csv(args.output / "source-paired-differences.csv", differences)
    graph_means = []
    for panel, content in replication.items():
        for method, points in content["curves"].items():
            for point in points:
                for evidence in point["evidence"]:
                    graph_means.append({"panel": panel, "method": method, "n": point["n"],
                        "k": content["k"], "density_m_over_n": content["density_m_over_n"],
                        "generator_seed": evidence["generator_seed"],
                        "graph_mean_fraction": evidence["graph_mean_fraction"]})
    write_csv(args.output / "source-graph-means.csv", graph_means)
    run_audit(args.qa_tools, "validate_figure.py", [Path(__file__), "--json"], args.output / "source-audit.json")
    print("Exported both complete-data figures with retained paired differences and condition-specific new-graph ranges.")


if __name__ == "__main__":
    main()
