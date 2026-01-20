#!/usr/bin/env python3
"""
Generate the main body dataset statistics table (tab:dataset-stats)
by aggregating data from the CSV files.

This script creates a LaTeX table showing the breakdown of papers by
time period (pre-LLM: 2020-2022, post-LLM: 2023-2025) for Computer Science papers.
"""

import pandas as pd
from pathlib import Path


def generate_dataset_stats_table():
    """Generate the dataset-stats table from CSV data."""

    # Read the Computer Science data
    data_dir = Path(__file__).parent.parent / "data" / "tables"
    cs_file = data_dir / "stats_computer_science.csv"

    df = pd.read_csv(cs_file)

    # Split by time period
    pre_llm = df[df['year'].isin([2020, 2021, 2022])]
    post_llm = df[df['year'].isin([2023, 2024, 2025])]

    # Calculate totals for each period
    pre_llm_review = pre_llm['Review'].sum()
    pre_llm_regular = pre_llm['Regular'].sum()
    pre_llm_total = pre_llm['Total'].sum()

    post_llm_review = post_llm['Review'].sum()
    post_llm_regular = post_llm['Regular'].sum()
    post_llm_total = post_llm['Total'].sum()

    # Overall totals
    total_review = df['Review'].sum()
    total_regular = df['Regular'].sum()
    total_total = df['Total'].sum()

    # Generate LaTeX table (no commas in numbers for consistency with other tables)
    latex_table = f"""\\begin{{table}}[t]
\\centering
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Category}} & \\textbf{{2020-2022}} & \\textbf{{2023-2025}} & \\textbf{{Total}} \\\\
\\midrule
Review Papers   & {pre_llm_review:,}   & {post_llm_review:,}   & {total_review:,}   \\\\
Regular Papers  & {pre_llm_regular:,} & {post_llm_regular:,} & {total_regular:,} \\\\
\\midrule
Total           & {pre_llm_total:,} & {post_llm_total:,} & {total_total:,} \\\\
\\bottomrule
\\end{{tabular}}}}
\\caption{{Dataset composition of computer science papers by paper type and time period.}}
\\label{{tab:dataset-stats}}
\\end{{table}}
"""

    # Save to file
    output_file = data_dir / "dataset_stats_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"✓ Generated dataset-stats table: {output_file}")
    print(f"\nSummary:")
    print(f"  Pre-LLM (2020-2022):")
    print(f"    Review:  {pre_llm_review:6,}  Regular: {pre_llm_regular:6,}  Total: {pre_llm_total:6,}")
    print(f"  Post-LLM (2023-2025):")
    print(f"    Review:  {post_llm_review:6,}  Regular: {post_llm_regular:6,}  Total: {post_llm_total:6,}")
    print(f"  Overall Total:")
    print(f"    Review:  {total_review:6,}  Regular: {total_regular:6,}  Total: {total_total:6,}")


if __name__ == "__main__":
    generate_dataset_stats_table()
