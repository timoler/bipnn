"""Matched-instance comparisons from the complete, audited experimental JSON."""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-tools", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    data = json.loads(args.summary.read_text(encoding="utf8"))
    if data["completed_cases"] != data["expected_cases"]:
        raise ValueError("Only render the full prespecified suite")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.titlesize": 7.5, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 75/25.4))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.76, wspace=0.38)
    definitions = [("weak", "Weak independence", "#376996", "o"),
                   ("strong", "Strong independence", "#B35D69", "s")]
    baselines = [("degree_greedy", "vs degree greedy"),
                 ("random_score_same_decoder", "vs random scores + same decoder")]
    sizes = data["config"]["sizes"]
    source = []
    for j, (ax, (baseline, title)) in enumerate(zip(axes, baselines)):
        extrema = [0.0]
        for mode_index, (mode, label, color, marker) in enumerate(definitions):
            means, deviations = [], []
            positions = []
            for x, n in enumerate(sizes):
                cases = [c for c in data["cases"] if c["definition"] == mode and c["n"] == n]
                values = []
                for index, c in enumerate(cases):
                    nn_mean = c["methods"]["hgnnp_pubo_repair"]["mean_size"]
                    reference_mean = c["methods"][baseline]["mean_size"]
                    value = 100 * (nn_mean / reference_mean - 1)
                    values.append(value)
                    # Offset is purely graphical; all case-level values are retained.
                    offset = (index-(len(cases)-1)/2) * 0.018 + (-0.055 if mode_index == 0 else 0.055)
                    ax.scatter(x+offset, value, s=10, color=color, alpha=0.45,
                               marker=marker, linewidths=0, zorder=2)
                    source.append({"panel": j+1, "case": c["name"], "definition": mode,
                                   "n": n, "baseline": baseline, "percent_change": value})
                center = st.mean(values)
                spread = st.stdev(values) if len(values) > 1 else 0.0
                means.append(center)
                deviations.append(spread)
                extrema.extend(values + [center-spread, center+spread])
                positions.append(x + (-0.055 if mode_index == 0 else 0.055))
            ax.errorbar(positions, means, yerr=deviations, color=color, marker=marker,
                        markersize=3.5, linewidth=1.1, elinewidth=0.8, capsize=2,
                        label=label, zorder=3)
        ax.axhline(0, color="#777777", linewidth=0.7, linestyle="--", zorder=1)
        padding = max(2, (max(extrema)-min(extrema))*0.15)
        ax.set_ylim(min(extrema)-padding, max(extrema)+padding)
        ax.set_xlim(-0.3, len(sizes)-0.7)
        ax.set_xticks(range(len(sizes)), [str(n) for n in sizes])
        ax.set_xlabel("Vertices")
        ax.set_ylabel("Change in solution size (%)")
        ax.set_title(title, pad=7)
        ax.text(-0.15, 1.14, chr(ord("a")+j), transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="left", va="bottom")
        ax.tick_params(direction="out", length=3, width=0.7)
    handles = [Line2D([], [], color=color, marker=marker, linewidth=1.1, markersize=3.5, label=label)
               for mode, label, color, marker in definitions]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2)
    fig.canvas.draw()
    stem = args.output / "paired-comparisons"
    require_matplotlib_panel_alignment(fig, json_out=str(stem)+".alignment.json",
                                      overlay_svg=str(stem)+".alignment.svg",
                                      tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                      require_panel_labels=True, strict=True)
    fig.savefig(str(stem)+".pdf")
    fig.savefig(str(stem)+".svg")
    fig.savefig(str(stem)+".png", dpi=400)
    plt.close(fig)
    (args.output / "source-data.json").write_text(json.dumps(source, indent=2), encoding="utf8")
    contract = {
        "question": "Does trained HGNN+ outperform degree greedy, and does training outperform random scores under the same decoder?",
        "archetype": "quantitative grid", "backend": "Python matplotlib",
        "dimensions_mm": [180, 75], "target": "graduate thesis experimental report",
        "panel_a": "matched comparison against a simple deterministic baseline",
        "panel_b": "matched control for the contribution of training rather than decoding",
        "observations": "all 48 definition-instance combinations; no outcomes excluded",
        "center": "mean across 8 configuration-instance combinations per size and definition, after averaging 3 seeds within each instance",
        "uncertainty": "descriptive sample SD across 8 case-level values; not a confidence interval or significance test",
        "dots": "all case-level paired percentage changes; horizontal offsets only for legibility",
        "statistics": "no p values; related density configurations may share edges",
        "training": "per-instance optimization, no cross-instance train/test claim",
        "source": "complete audited summary and source-data.json",
    }
    (args.output / "figure-contract.json").write_text(json.dumps(contract, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
