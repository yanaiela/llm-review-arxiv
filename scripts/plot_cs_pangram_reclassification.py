"""Plot CS Pangram full-text reclassification rates by year.

Reads the Pangram full-text reclassification JSON for CS and builds a
year-split bar chart (review vs non-review) similar to the
`adjusted_alpha_by_category_and_year_pangram` visualization.

Classification rule: any `prediction` other than "Unlikely AI" is treated as AI.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REGULAR_LABEL = "Non-Review Papers"
REVIEW_LABEL = "Review Papers"
REGULAR_COLOR = "#4ECDC4"
REVIEW_COLOR = "#FF6B6B"


def extract_year(arxiv_id: str) -> int:
    """Convert an arXiv identifier (yymm.number) into a calendar year."""

    prefix = arxiv_id.split(".")[0][:4]
    yy = int(prefix[:2])
    return 2000 + yy


def load_reclass_records(path: Path) -> list[dict]:
    """Load reclassification JSON and normalize the fields we need.

    Uses prediction-based classification: prediction != "Unlikely AI" -> AI.
    """

    raw = json.loads(path.read_text())
    records: list[dict] = []

    for item in raw:
        reclass = item.get("reclassified_fulltext") or {}
        prediction = reclass.get("prediction")
        if not prediction or not isinstance(prediction, str):
            continue

        pred_lower = prediction.strip().lower()
        ai_label = 0 if pred_lower == "unlikely ai" else 1

        records.append(
            {
                "year": extract_year(str(item["arxiv_id"])),
                "paper_type": item.get("paper_type", "regular"),
                "ai_label": ai_label,
                "arxiv_id": str(item["arxiv_id"]),
            }
        )

    return records


def load_original_counts(path: Path) -> dict[int, dict[str, dict]]:
    """Load the full Pangram detection results and count totals/flagged by year/paper type."""

    raw = json.loads(path.read_text())
    counts: dict[int, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "flagged": 0}))

    for item in raw:
        year = int(item.get("year"))
        paper_type = item.get("paper_type", "regular")
        prediction = (item.get("pangram_prediction") or {}).get("prediction")
        pred_lower = prediction.strip().lower() if isinstance(prediction, str) else ""
        ai_label = 0 if pred_lower == "unlikely ai" else 1

        counts[year][paper_type]["total"] += 1
        counts[year][paper_type]["flagged"] += ai_label

    return counts


def compute_rates(records: list[dict]) -> dict[int, dict[str, dict]]:
    """Aggregate AI classification rate (prediction-based) by year and paper type."""

    stats: dict[int, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"values": []}))

    for rec in records:
        year = rec["year"]
        paper_type = rec["paper_type"]
        stats[year][paper_type]["values"].append(rec["ai_label"])

    return stats


def summarize_rates(stats: dict[int, dict[str, dict]]) -> dict[int, dict[str, dict]]:
    """Add mean AI classification rate and its standard error per year and paper type."""

    summary: dict[int, dict[str, dict]] = defaultdict(dict)

    for year, year_data in stats.items():
        for paper_type, payload in year_data.items():
            values = payload["values"]
            n = len(values)
            if n == 0:
                continue

            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n if n > 0 else 0.0
            std = math.sqrt(variance)
            sem = std / math.sqrt(n) if n > 0 else 0.0
            summary[year][paper_type] = {"rate": mean, "sem": sem, "n": n}

    return summary


def compute_overall_adjusted(adjusted_summary: dict[int, dict[str, dict]]) -> dict[str, dict]:
    """Compute overall adjusted rates and SEM per paper type across all years.

    Sem is propagated assuming independence: var = sum((sem*total)^2) / (sum(total)^2).
    """

    totals: dict[str, float] = defaultdict(float)
    ai_estimates: dict[str, float] = defaultdict(float)
    var_terms: dict[str, float] = defaultdict(float)

    for year_data in adjusted_summary.values():
        for paper_type, entry in year_data.items():
            total = entry["total"]
            rate = entry["rate"]
            sem = entry["sem"]
            ai_est = rate * total
            totals[paper_type] += total
            ai_estimates[paper_type] += ai_est
            var_terms[paper_type] += (sem * total) ** 2

    overall: dict[str, dict] = {}
    for paper_type, total in totals.items():
        if total == 0:
            continue
        ai_est = ai_estimates[paper_type]
        rate = ai_est / total
        sem = math.sqrt(var_terms[paper_type]) / total if total > 0 else 0.0
        overall[paper_type] = {"rate": rate, "sem": sem, "total": total, "ai_est": ai_est}

    return overall


def adjust_with_original(
    reclass_summary: dict[int, dict[str, dict]], original_counts: dict[int, dict[str, dict]]
) -> dict[int, dict[str, dict]]:
    """Adjust rates using original abstract-level counts.

    For each (year, paper_type):
      - total = all papers in original counts
      - flagged = papers predicted AI by abstract (prediction != Unlikely AI)
      - if reclass sample exists: use sample rate p to estimate AI_count = flagged * p
        adjusted_rate = AI_count / total; sem scales accordingly.
      - else: fall back to abstract-level rate = flagged / total.
    """

    adjusted: dict[int, dict[str, dict]] = defaultdict(dict)

    for year, year_data in original_counts.items():
        for paper_type, counts in year_data.items():
            total = counts["total"]
            flagged = counts["flagged"]
            if total == 0:
                continue

            orig_rate = flagged / total
            if year in reclass_summary and paper_type in reclass_summary[year]:
                sample = reclass_summary[year][paper_type]
                p = sample["rate"]
                sem_sample = sample["sem"]
                n_sample = sample["n"]
                ai_est = flagged * p
                rate = ai_est / total
                sem = (flagged / total) * sem_sample
            else:
                p = orig_rate
                sem_sample = 0.0
                n_sample = 0
                rate = orig_rate
                sem = 0.0

            adjusted[year][paper_type] = {
                "rate": rate,
                "sem": sem,
                "n_sample": n_sample,
                "flagged": flagged,
                "total": total,
                "orig_rate": orig_rate,
                "p_reclass": p,
                "sem_sample": sem_sample,
            }

    return adjusted


def filter_years(summary: dict[int, dict[str, dict]], min_year: int = 2023) -> dict[int, dict[str, dict]]:
    """Return a copy of summary limited to years >= min_year."""

    return {year: data for year, data in summary.items() if year >= min_year}


def plot(summary: dict[int, dict[str, dict]], output_dir: Path, basename: str) -> None:
    years = sorted(summary.keys())
    if not years:
        raise ValueError("No data available to plot.")

    fig_width = max(10, 3.0 * len(years))
    fig, axes = plt.subplots(1, len(years), figsize=(fig_width, 5), sharey=True)
    if hasattr(axes, "ravel"):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    max_y = 0.0
    for year in years:
        for pt_data in summary[year].values():
            y_val = pt_data["rate"] + pt_data.get("sem", 0.0)
            max_y = max(max_y, y_val)

    y_limit = max_y * 1.12 if max_y > 0 else 1.0

    for ax, year in zip(axes, years):
        year_data = summary.get(year, {})
        categories = ["regular", "review"]
        x_positions = [0, 0.7]
        width = 0.55

        heights = [year_data.get(cat, {}).get("rate", 0.0) for cat in categories]
        errors = [year_data.get(cat, {}).get("sem", 0.0) for cat in categories]

        bars = ax.bar(
            [x_positions[0]],
            [heights[0]],
            width,
            yerr=[errors[0]],
            label=REGULAR_LABEL,
            color=REGULAR_COLOR,
            alpha=0.8,
            capsize=5,
        )

        bars2 = ax.bar(
            [x_positions[1]],
            [heights[1]],
            width,
            yerr=[errors[1]],
            label=REVIEW_LABEL,
            color=REVIEW_COLOR,
            alpha=0.8,
            capsize=5,
        )

        ax.set_title(str(year), fontsize=16, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Non-Review", "Review"], fontsize=12)
        ax.set_ylim(0, y_limit)
        if ax is axes[0]:
            ax.set_ylabel("Estimated LLM Fraction", fontsize=14)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.legend(fontsize=12)
        ax.grid(alpha=0.3, axis="y")

        for bar in list(bars) + list(bars2):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

    fig.suptitle("CS Full-Text Pangram Rates by Year", fontsize=18, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CS Pangram full-text reclassification rates by year (review vs regular)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/results/cs/pangram_fulltext_reclassification.json"),
        help="Path to pangram full-text reclassification JSON file.",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("data/results/cs/pangram_detection_results.json"),
        help="Path to original Pangram detection results JSON (abstract-level).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/figures"),
        help="Directory to write the output figures.",
    )
    parser.add_argument(
        "--basename",
        default="cs_pangram_fulltext_reclassification_by_year",
        help="Base filename (without extension) for the saved figures.",
    )
    args = parser.parse_args()

    reclass_records = load_reclass_records(args.input)
    original_counts = load_original_counts(args.original)

    # Sample (reclass) rates
    stats = compute_rates(reclass_records)
    reclass_summary = summarize_rates(stats)

    # Adjust using original abstract-level counts
    adjusted_summary = adjust_with_original(reclass_summary, original_counts)

    # Filter out early years (pre-2023)
    adjusted_summary = filter_years(adjusted_summary, min_year=2023)

    overall = compute_overall_adjusted(adjusted_summary)

    # Print overall percentages to stdout
    print("Adjusted AI classification rate (prediction != 'Unlikely AI', weighted by original counts):")
    for paper_type in sorted(overall.keys()):
        entry = overall[paper_type]
        pct = entry["rate"] * 100
        sem_pct = entry["sem"] * 100
        print(f"  {paper_type:<7} N={int(entry['total']):5d}  {pct:5.1f}%  (sem {sem_pct:4.1f}%)")

    plot(adjusted_summary, args.output_dir, args.basename)


if __name__ == "__main__":
    main()
