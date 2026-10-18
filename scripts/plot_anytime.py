"""Publication figure for complete, audited strict-cutoff anytime results.

The two structures, largest-n time sweep and full size sweep are predeclared.
Usage: python scripts/plot_anytime.py --qa-tools <QA tools directory>
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = [(3, 2), (5, 4)]
SIZES = [500, 1000, 2000]
BUDGETS = [5, 15, 30]
METHODS = ["bipnn_fast_anytime", "random_restart_same_decoder", "scip_milp_greedy_start", "degree_greedy"]
STYLE = {
    "bipnn_fast_anytime": {"label": "BIPNN (adapted)", "color": "#286F8E", "marker": "o", "linestyle": "-"},
    "random_restart_same_decoder": {"label": "Random restart", "color": "#8666A5", "marker": "D", "linestyle": "-."},
    "scip_milp_greedy_start": {"label": "SCIP", "color": "#BB742D", "marker": "s", "linestyle": "--"},
    "degree_greedy": {"label": "Degree greedy", "color": "#777777", "marker": "^", "linestyle": ":"},
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def checked_summary(path):
    raw = path.read_bytes()
    summary = json.loads(raw)
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if summary.get("complete") is not True:
        raise ValueError("Final figures require a complete, audited summary")
    if summary["manifest_binding"] != manifest["binding"]:
        raise ValueError("Summary and experiment manifest do not match")
    if len(summary["case_summary"]) != 36:
        raise ValueError("Expected the complete 36-row synthetic and real-data summary")
    if list(manifest["plan"]["checkpoints_seconds"]) != BUDGETS:
        raise ValueError("This figure is bound to the predeclared 5/15/30-second panel map")
    if list(manifest["plan"]["model_seeds"]) != [0, 1, 2]:
        raise ValueError("The figure requires the three predeclared model seeds")
    selected = [row for row in summary["case_summary"] if row["scope"] == "synthetic_new"]
    expected = {(n, k, density, budget) for n in SIZES for k, density in STRUCTURES for budget in BUDGETS}
    indexed = {}
    for row in selected:
        generator = row["generator"]
        key = (row["n"], generator["k"], generator["density_m_over_n"], row["budget_seconds"])
        if row["definition"] != "weak" or key in indexed:
            raise ValueError("Unexpected definition or duplicate synthetic size/structure/budget")
        indexed[key] = row
        for sizes_key, mean_key in (("neural_sizes", "neural_mean_size"), ("random_sizes", "random_mean_size")):
            values = row[sizes_key]
            if len(values) != 3 or any(not math.isfinite(v) or v < 0 or v > row["n"] for v in values):
                raise ValueError("Missing or invalid per-seed feasible solution sizes")
            if not math.isclose(float(np.mean(values)), row[mean_key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Summary mean differs from its retained seed values")
        for key_name in ("scip_size", "degree_size"):
            if not 0 <= row[key_name] <= row["n"]:
                raise ValueError("Baseline solution size outside the explicit node count")
    if set(indexed) != expected or len(selected) != 18:
        raise ValueError("The figure requires all 18 synthetic case/budget rows, without exclusions")
    for n in SIZES:
        for k, density in STRUCTURES:
            ordered = [indexed[(n, k, density, budget)] for budget in BUDGETS]
            if len({row["degree_size"] for row in ordered}) != 1:
                raise ValueError("The deterministic greedy baseline changed between cutoffs")
            for key_name in ("neural_sizes", "random_sizes"):
                array = np.asarray([row[key_name] for row in ordered])
                if np.any(np.diff(array, axis=0) < 0):
                    raise ValueError("A prefix-best neural/random trajectory decreased")
            # SCIP deliberately has no monotonicity assertion: budgets are independent runs.
    return summary, manifest, indexed, hashlib.sha256(raw).hexdigest()


def series(row, method):
    n = row["n"]
    if method == "bipnn_fast_anytime":
        values = np.asarray(row["neural_sizes"], dtype=float) / n
    elif method == "random_restart_same_decoder":
        values = np.asarray(row["random_sizes"], dtype=float) / n
    else:
        values = np.asarray([row["scip_size"] if method == "scip_milp_greedy_start" else row["degree_size"]], dtype=float) / n
    return {"mean": float(values.mean()), "minimum": float(values.min()),
            "maximum": float(values.max()), "seed_values": values.tolist()}


def run_audit(tools, filename, arguments, output):
    result = subprocess.run([sys.executable, str(tools / filename), *map(str, arguments)],
                            capture_output=True, text=True, encoding="utf8",
                            env={**os.environ, "PYTHONUTF8": "1"})
    output.write_text(result.stdout + result.stderr, encoding="utf8")
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"Figure audit failed: {filename}; see {output.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "results/fast-anytime-20261018/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/fast-anytime-20261018/figures")
    parser.add_argument("--qa-tools", type=Path, required=True)
    args = parser.parse_args()
    summary, manifest, indexed, source_hash = checked_summary(args.summary)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.65, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    width_mm, height_mm = 180, 132
    fig, axes = plt.subplots(2, 2, figsize=(width_mm / 25.4, height_mm / 25.4), sharey="row")
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.15, top=0.87, wspace=0.28, hspace=0.52)
    panel_data = {}
    for row_index, (k, density) in enumerate(STRUCTURES):
        lower_values, upper_values = [], []
        for column in (0, 1):
            axis = axes[row_index, column]
            panel = chr(ord("a") + row_index * 2 + column)
            xvalues = BUDGETS if column == 0 else SIZES
            selected = ([indexed[(2000, k, density, budget)] for budget in BUDGETS] if column == 0 else
                        [indexed[(n, k, density, 30)] for n in SIZES])
            curves = {}
            for method in METHODS:
                statistics = [series(row, method) for row in selected]
                means = np.asarray([item["mean"] for item in statistics])
                lows = np.asarray([item["minimum"] for item in statistics])
                highs = np.asarray([item["maximum"] for item in statistics])
                lower_values.extend(lows.tolist())
                upper_values.extend(highs.tolist())
                style = STYLE[method]
                common = {"color": style["color"], "marker": style["marker"], "linestyle": style["linestyle"],
                          "linewidth": 1.15 if method == "bipnn_fast_anytime" else 0.9,
                          "markersize": 3.5, "markeredgewidth": 0.55}
                if method in ("bipnn_fast_anytime", "random_restart_same_decoder"):
                    axis.errorbar(xvalues, means, yerr=np.vstack((means-lows, highs-means)),
                                  capsize=1.9, capthick=0.65, elinewidth=0.65, **common)
                else:
                    axis.plot(xvalues, means, **common)
                curves[method] = {"x": xvalues, "statistics": statistics,
                                  "case_ids": [row["case_id"] for row in selected],
                                  "budgets_seconds": [row["budget_seconds"] for row in selected]}
            axis.set_xticks(xvalues)
            if column == 0:
                axis.set_xlabel("Time cutoff (s)", labelpad=4)
                axis.set_xlim(3, 32)
                axis.set_ylabel("Feasible set size / n", labelpad=5)
                suffix = "n = 2,000"
            else:
                axis.set_xlabel("Nodes n", labelpad=4)
                axis.set_xlim(380, 2120)
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:,.0f}"))
                suffix = "30-second cutoff"
            axis.set_title(f"k = {k}, m/n = {density}  |  {suffix}", fontsize=7.5, pad=8)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.2f}"))
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#E7E7E7", linewidth=0.5)
            axis.annotate(panel, xy=(0, 1), xycoords="axes fraction", xytext=(-23, 8),
                          textcoords="offset points", fontsize=9, fontweight="bold",
                          ha="left", va="bottom", annotation_clip=False)
            panel_data[panel] = {"structure": {"k": k, "density_m_over_n": density},
                                 "dimension": "time" if column == 0 else "scale", "curves": curves}
        # A proportion need not start at zero in a point/line chart. Both panels
        # in the same structure row share limits and include every seed extreme.
        pad = max(0.015, (max(upper_values)-min(lower_values))*0.1)
        low = max(0., math.floor((min(lower_values)-pad)*20)/20)
        high = min(1., math.ceil((max(upper_values)+pad)*20)/20)
        if high <= low:
            raise ValueError("Degenerate plotting range")
        axes[row_index, 0].set_ylim(low, high)
    handles = [Line2D([], [], label=STYLE[m]["label"], color=STYLE[m]["color"],
                       marker=STYLE[m]["marker"], linestyle=STYLE[m]["linestyle"],
                       linewidth=1, markersize=3.5) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.985),
               ncol=4, frameon=False, handlelength=2.1, handletextpad=0.6, columnspacing=1.5)
    fig.text(0.54, 0.035, "Means and seed ranges; one synthetic graph per size and structure.",
             fontsize=7, ha="center", va="bottom")
    stem = args.output / "strict-cutoff-quality"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=axes.flatten().tolist(), panel_ids=["a", "b", "c", "d"],
        json_out=str(stem) + ".alignment.json", overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".svg")
    fig.savefig(str(stem) + ".png", dpi=400)
    plt.close(fig)
    write_json(args.output / "source-data.json", {
        "summary_source": "../summary.json", "summary_sha256": source_hash,
        "manifest_binding": manifest["binding"], "excluded_summary_rows": 0,
        "all_case_summary": summary["case_summary"], "panels": panel_data,
        "scope_note": "All 36 summary rows retained; plotted subset follows the predeclared panel map. Real cases are tabulated in the report.",
        "spread": "Minimum to maximum of three technical seed trajectories; no confidence intervals or p-values"})
    run_audit(args.qa_tools, "validate_figure.py", [Path(__file__), "--json"], args.output / "source-audit.json")
    run_audit(args.qa_tools, "audit_pdf_text.py", [str(stem) + ".pdf", "--min-pt", "5", "--json"],
              args.output / "font-audit.json")
    run_audit(args.qa_tools, "audit_figure_collisions.py",
              [str(stem) + ".pdf", "--json-out", str(stem) + ".collision-audit.json",
               "--overlay-pdf", str(stem) + ".collision-audit.pdf"], args.output / "collision-audit.txt")
    print(f"Exported {width_mm} x {height_mm} mm PDF/SVG/400dpiPNG from complete strict-cutoff data.")


if __name__ == "__main__":
    main()
