"""Plot every dense/indexed loss-kernel microbenchmark case, with rendered QA.

Example: python scripts/plot_loss_kernel.py --qa-tools <QA tools directory>
The QA helper location is supplied at run time and is not part of figure text.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def run_audit(tools, filename, arguments, output):
    result = subprocess.run([sys.executable, str(tools / filename), *map(str, arguments)],
                            capture_output=True, text=True, encoding="utf8",
                            env={**os.environ, "PYTHONUTF8": "1"})
    output.write_text(result.stdout + result.stderr, encoding="utf8")
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"Rendered/source audit failed: {filename}; see {output.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results/loss-kernel-20261018/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/loss-kernel-20261018/figures")
    parser.add_argument("--qa-tools", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    source_bytes = args.input.read_bytes()
    summary = json.loads(source_bytes)
    records = summary["records"]
    expected_keys = {(n, k, mode) for n in (200, 500, 1000) for k in (3, 5)
                     for mode in ("weak", "strong")}
    actual_keys = {(r["n"], r["k"], r["definition"]) for r in records}
    if summary["number_failed"] or actual_keys != expected_keys or len(records) != 12:
        raise ValueError("Figure requires all 12 successful, unique benchmark combinations")
    rows = []
    for record in records:
        dense = np.asarray(record["dense"]["forward_backward_seconds_per_iteration"], dtype=float)
        indexed = np.asarray(record["indexed"]["forward_backward_seconds_per_iteration"], dtype=float)
        if np.any(dense <= 0) or np.any(indexed <= 0):
            raise ValueError("Log-ratio inputs must be strictly positive")
        memory_dense = record["dense"]["incremental_peak_allocated_bytes"]
        memory_indexed = record["indexed"]["incremental_peak_allocated_bytes"]
        if min(memory_dense, memory_indexed) <= 0:
            raise ValueError("Additional peak memory must be positive for a ratio")
        rows.append({"n": record["n"], "k": record["k"], "definition": record["definition"],
                     "time_ratio": float(np.median(dense) / np.median(indexed)),
                     "time_range_lower": float(dense.min() / indexed.max()),
                     "time_range_upper": float(dense.max() / indexed.min()),
                     "memory_ratio": memory_dense / memory_indexed,
                     "dense_time_blocks_seconds": dense.tolist(),
                     "indexed_time_blocks_seconds": indexed.tolist(),
                     "dense_additional_peak_bytes": memory_dense,
                     "indexed_additional_peak_bytes": memory_indexed})
    source_data = {"source": "../summary.json", "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                   "aggregation": "Ratio of the medians of five block means, five iterations per block",
                   "interval": "Observed range envelope: min(dense)/max(indexed) to max(dense)/min(indexed); not a confidence interval",
                   "excluded_rows": 0, "plot_rows": rows, "raw_records": records}
    write_json(args.output / "source-data.json", source_data)

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.linewidth": 0.65, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    width_mm, height_mm = 180, 88
    fig, axes = plt.subplots(1, 2, figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.23, top=0.805, wspace=0.34)
    combinations = [(n, k) for n in (200, 500, 1000) for k in (3, 5)]
    palette = {"weak": "#286F8E", "strong": "#B46A2B"}
    markers = {"weak": "o", "strong": "s"}
    by_key = {(r["n"], r["k"], r["definition"]): r for r in rows}
    for mode, offset in (("weak", -0.13), ("strong", 0.13)):
        ordered = [by_key[(n, k, mode)] for n, k in combinations]
        x = np.arange(len(combinations)) + offset
        center = np.asarray([r["time_ratio"] for r in ordered])
        low = np.asarray([r["time_range_lower"] for r in ordered])
        high = np.asarray([r["time_range_upper"] for r in ordered])
        if np.any(low > center) or np.any(high < center):
            raise AssertionError("Timing center lies outside its descriptive range")
        axes[0].errorbar(x, center, yerr=np.vstack((center-low, high-center)),
                         fmt=markers[mode], markersize=4.1, color=palette[mode],
                         elinewidth=0.8, capsize=2.1, capthick=0.8, linestyle="none")
        # Memory is a single peak observation, not a seed/block mean; no interval.
        axes[1].plot(x, [r["memory_ratio"] for r in ordered], markers[mode],
                     markersize=4.1, color=palette[mode], linestyle="none")
    axes[0].set_yscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[0].set_ylim(0.125, 16)
    axes[1].set_ylim(8, 256)
    axes[0].yaxis.set_major_locator(FixedLocator([0.125, 0.25, 0.5, 1, 2, 4, 8, 16]))
    axes[1].yaxis.set_major_locator(FixedLocator([8, 16, 32, 64, 128, 256]))
    for i, axis in enumerate(axes):
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_xlim(-0.55, 5.55)
        axis.set_xticks(range(6), [f"{n}\nk={k}" for n, k in combinations])
        axis.set_xlabel("Nodes n / hyperedge size k", labelpad=5)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        for boundary in (1.5, 3.5):
            axis.axvline(boundary, color="#D8D8D8", linewidth=0.5, linestyle=(0, (2, 3)))
        axis.annotate(chr(ord("a") + i), xy=(0, 1), xycoords="axes fraction",
                      xytext=(-25, 8), textcoords="offset points", fontsize=9,
                      fontweight="bold", ha="left", va="bottom", annotation_clip=False)
    axes[0].axhline(1, color="#606060", linewidth=0.8, linestyle=(0, (4, 3)))
    axes[0].set_ylabel("Dense / indexed time")
    axes[1].set_ylabel("Dense / indexed additional peak memory")
    axes[0].set_title("Forward + backward time", fontsize=8, pad=8)
    axes[1].set_title("Additional peak memory", fontsize=8, pad=8)
    legend_handles = [Line2D([], [], marker=markers[mode], color=palette[mode],
                             linestyle="none", markersize=4.3,
                             label=label) for mode, label in
                      (("weak", "Weak independent set"), ("strong", "Strong independent set"))]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.54, 0.985),
               ncol=2, frameon=False, handletextpad=0.5, columnspacing=2.5)
    fig.text(0.54, 0.028, "Ratios above 1 favor indexed evaluation; timing ranges are descriptive.",
             ha="center", va="bottom", fontsize=7)
    stem = args.output / "loss-kernel-comparison"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=["a", "b"], json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".svg")
    fig.savefig(str(stem) + ".png", dpi=400)
    plt.close(fig)
    run_audit(args.qa_tools, "validate_figure.py", [Path(__file__), "--json"],
              args.output / "source-audit.json")
    run_audit(args.qa_tools, "audit_pdf_text.py", [str(stem) + ".pdf", "--min-pt", "5", "--json"],
              args.output / "font-audit.json")
    run_audit(args.qa_tools, "audit_figure_collisions.py",
              [str(stem) + ".pdf", "--json-out", str(stem) + ".collision-audit.json",
               "--overlay-pdf", str(stem) + ".collision-audit.pdf"],
              args.output / "collision-audit.txt")
    print(f"Exported {width_mm} x {height_mm} mm PDF/SVG and 400 dpi PNG; all 12 cases retained.")


if __name__ == "__main__":
    main()
