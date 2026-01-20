"""
Script to generate LaTeX tables for the appendix similar to those in the paper.
Reads data from processed metadata and statistical analysis files to create:
1. Dataset statistics tables by category (CS, Math, Stats, Physics)
2. Paper counts by year, broken down by review vs regular papers
3. Estimated totals extrapolated to full arXiv population
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple


def load_metadata(category: str) -> pd.DataFrame:
    """
    Load paper metadata for a given category.

    Args:
        category: The arXiv category (e.g., 'cs', 'math', 'stat', 'physics')

    Returns:
        DataFrame with paper metadata
    """
    data_dir = Path(__file__).parent.parent / 'data'
    metadata_path = data_dir / 'processed' / category / 'paper_metadata.csv'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Parse submission_date to extract year
    df['submission_date'] = pd.to_datetime(df['submission_date'], errors='coerce')
    df['year'] = df['submission_date'].dt.year

    return df


def load_classifications(category: str) -> pd.DataFrame:
    """
    Load paper classifications for a given category.

    Args:
        category: The arXiv category (e.g., 'cs', 'math', 'stat', 'physics')

    Returns:
        DataFrame with paper classifications (review vs regular)
    """
    data_dir = Path(__file__).parent.parent / 'data'
    classifications_path = data_dir / 'processed' / category / 'paper_classifications.csv'

    if not classifications_path.exists():
        raise FileNotFoundError(f"Classifications file not found: {classifications_path}")

    return pd.read_csv(classifications_path)


def merge_data(metadata: pd.DataFrame, classifications: pd.DataFrame) -> pd.DataFrame:
    """
    Merge metadata with classifications.

    Args:
        metadata: DataFrame with paper metadata
        classifications: DataFrame with paper classifications

    Returns:
        Merged DataFrame
    """
    return metadata.merge(classifications, on='arxiv_id', how='left')


def compute_year_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics by year and paper type.

    Args:
        df: DataFrame with merged metadata and classifications

    Returns:
        DataFrame with year-level statistics
    """
    # Filter to valid years (2020-2025)
    df_filtered = df[df['year'].between(2020, 2025)].copy()

    # Create paper type column (review vs regular)
    df_filtered['is_review'] = df_filtered['predicted_type'] == 'review'

    # Count papers by year and type
    stats = df_filtered.groupby(['year', 'is_review']).size().unstack(fill_value=0)
    stats.columns = ['Regular', 'Review']

    # Reorder columns
    if 'Review' in stats.columns and 'Regular' in stats.columns:
        stats = stats[['Review', 'Regular']]

    # Add total column
    stats['Total'] = stats.sum(axis=1)

    # Calculate percentages
    stats['Review %'] = (stats['Review'] / stats['Total'] * 100).round(1)
    stats['Regular %'] = (stats['Regular'] / stats['Total'] * 100).round(1)

    return stats


def estimate_total_arxiv_papers(sample_stats: pd.DataFrame,
                                 category: str,
                                 sampling_ratio: float = 0.1) -> pd.DataFrame:
    """
    Estimate total arXiv papers based on sample statistics.

    Args:
        sample_stats: DataFrame with sample statistics
        category: The arXiv category
        sampling_ratio: Ratio of sampled papers to total papers

    Returns:
        DataFrame with estimated totals
    """
    # This is a simplified estimation - you may want to adjust based on actual sampling
    estimated = sample_stats.copy()
    estimated['Review Est. Total'] = (estimated['Review'] / sampling_ratio).astype(int)
    estimated['Regular Est. Total'] = (estimated['Regular'] / sampling_ratio).astype(int)
    estimated['Total Est. Total'] = estimated['Review Est. Total'] + estimated['Regular Est. Total']

    return estimated


def generate_latex_table(stats: pd.DataFrame,
                          category_name: str,
                          include_estimates: bool = False) -> str:
    """
    Generate LaTeX table code for the statistics.

    Args:
        stats: DataFrame with statistics
        category_name: Name of the category for the caption
        include_estimates: Whether to include estimated totals

    Returns:
        LaTeX table code as string
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\resizebox{\\columnwidth}{!}{%")

    if include_estimates:
        latex.append("\\begin{tabular}{lrrrrrr}")
        latex.append("\\toprule")
        latex.append("\\textbf{Year} & \\multicolumn{3}{c}{\\textbf{Review Papers}} & \\multicolumn{3}{c}{\\textbf{Regular Papers}} \\\\")
        latex.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
        latex.append(" & \\textbf{Sample} & \\textbf{Est. Total} & \\textbf{\\%} & \\textbf{Sample} & \\textbf{Est. Total} & \\textbf{\\%} \\\\")
    else:
        latex.append("\\begin{tabular}{lrrrr}")
        latex.append("\\toprule")
        latex.append("\\textbf{Year} & \\textbf{Review} & \\textbf{Regular} & \\textbf{Total} & \\textbf{Review \\%} \\\\")

    latex.append("\\midrule")

    # Add data rows
    total_review = 0
    total_regular = 0
    total_review_est = 0
    total_regular_est = 0

    for year in sorted(stats.index):
        row = stats.loc[year]
        total_review += row['Review']
        total_regular += row['Regular']

        if include_estimates and 'Review Est. Total' in row:
            total_review_est += row['Review Est. Total']
            total_regular_est += row['Regular Est. Total']
            latex.append(
                f"{int(year)} & {int(row['Review']):,} & {int(row['Review Est. Total']):,} & "
                f"{row['Review %']:.1f}\\% & {int(row['Regular']):,} & "
                f"{int(row['Regular Est. Total']):,} & {row['Regular %']:.1f}\\% \\\\"
            )
        else:
            latex.append(
                f"{int(year)} & {int(row['Review']):,} & {int(row['Regular']):,} & "
                f"{int(row['Total']):,} & {row['Review %']:.1f}\\% \\\\"
            )

    # Add totals row
    latex.append("\\midrule")
    if include_estimates and total_review_est > 0:
        total_pct = (total_review / (total_review + total_regular) * 100)
        latex.append(
            f"\\textbf{{Total}} & \\textbf{{{total_review:,}}} & \\textbf{{{total_review_est:,}}} & "
            f"\\textbf{{{total_pct:.1f}\\%}} & \\textbf{{{total_regular:,}}} & "
            f"\\textbf{{{total_regular_est:,}}} & \\textbf{{{100-total_pct:.1f}\\%}} \\\\"
        )
    else:
        total = total_review + total_regular
        total_pct = (total_review / total * 100) if total > 0 else 0
        latex.append(
            f"\\textbf{{Total}} & \\textbf{{{total_review:,}}} & \\textbf{{{total_regular:,}}} & "
            f"\\textbf{{{total:,}}} & \\textbf{{{total_pct:.1f}\\%}} \\\\"
        )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")

    # Add caption
    if include_estimates:
        latex.append(
            f"\\caption{{Number of {category_name} papers per year by paper type. "
            f"Sample shows our dataset; Est. Total shows extrapolated estimates for all arXiv papers in this category.}}"
        )
    else:
        latex.append(
            f"\\caption{{Dataset composition of {category_name} papers by year and paper type.}}"
        )

    latex.append(f"\\label{{tab:papers-{category_name.lower().replace(' ', '-')}-stats}}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_summary_table(all_stats: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a summary table across all categories.

    Args:
        all_stats: Dictionary mapping category names to their statistics DataFrames

    Returns:
        LaTeX table code as string
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\resizebox{\\columnwidth}{!}{%")
    latex.append("\\begin{tabular}{lrrrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Category} & \\textbf{Review} & \\textbf{Regular} & \\textbf{Total} & \\textbf{Review \\%} \\\\")
    latex.append("\\midrule")

    grand_total_review = 0
    grand_total_regular = 0

    for category, stats in all_stats.items():
        total_review = stats['Review'].sum()
        total_regular = stats['Regular'].sum()
        total = total_review + total_regular
        review_pct = (total_review / total * 100) if total > 0 else 0

        grand_total_review += total_review
        grand_total_regular += total_regular

        latex.append(
            f"{category} & {int(total_review):,} & {int(total_regular):,} & "
            f"{int(total):,} & {review_pct:.1f}\\% \\\\"
        )

    latex.append("\\midrule")
    grand_total = grand_total_review + grand_total_regular
    grand_review_pct = (grand_total_review / grand_total * 100) if grand_total > 0 else 0
    latex.append(
        f"\\textbf{{Total}} & \\textbf{{{grand_total_review:,}}} & \\textbf{{{grand_total_regular:,}}} & "
        f"\\textbf{{{grand_total:,}}} & \\textbf{{{grand_review_pct:.1f}\\%}} \\\\"
    )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")
    latex.append("\\caption{Summary of dataset composition across all categories.}")
    latex.append("\\label{tab:dataset-summary}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    """Main function to generate all tables."""
    categories = {
        'cs': 'Computer Science',
        'math': 'Mathematics',
        'stat': 'Statistics',
        'physics': 'Physics'
    }

    output_dir = Path(__file__).parent.parent / 'data' / 'tables'
    output_dir.mkdir(exist_ok=True, parents=True)

    all_stats = {}
    all_latex = []

    print("=" * 80)
    print("GENERATING APPENDIX TABLES")
    print("=" * 80)

    for category_key, category_name in categories.items():
        print(f"\n\nProcessing {category_name}...")
        print("-" * 80)

        try:
            # Load data
            metadata = load_metadata(category_key)
            classifications = load_classifications(category_key)

            # Merge data
            df = merge_data(metadata, classifications)

            # Compute statistics
            stats = compute_year_statistics(df)
            all_stats[category_name] = stats

            # Print statistics
            print(f"\n{category_name} Statistics:")
            print(stats.to_string())

            # Generate LaTeX table
            latex_table = generate_latex_table(stats, category_name, include_estimates=False)
            all_latex.append(f"\n% {category_name}\n{latex_table}\n")

            # Save individual table
            table_file = output_dir / f"table_{category_key}.tex"
            with open(table_file, 'w') as f:
                f.write(latex_table)
            print(f"\nSaved LaTeX table to: {table_file}")

        except Exception as e:
            print(f"Error processing {category_name}: {e}")
            continue

    # Generate summary table
    if all_stats:
        print("\n\n" + "=" * 80)
        print("SUMMARY ACROSS ALL CATEGORIES")
        print("=" * 80)
        summary_latex = generate_summary_table(all_stats)
        all_latex.insert(0, f"% Summary Table\n{summary_latex}\n")

        summary_file = output_dir / "table_summary.tex"
        with open(summary_file, 'w') as f:
            f.write(summary_latex)
        print(f"\nSaved summary table to: {summary_file}")

    # Save all tables to a single file
    all_tables_file = output_dir / "all_tables.tex"
    with open(all_tables_file, 'w') as f:
        f.write("% Auto-generated tables for appendix\n")
        f.write("% Generated by scripts/generate_appendix_tables.py\n\n")
        f.write("\n".join(all_latex))

    print("\n" + "=" * 80)
    print(f"All tables saved to: {all_tables_file}")
    print("=" * 80)

    # Also save statistics as CSV for reference
    for category_name, stats in all_stats.items():
        csv_file = output_dir / f"stats_{category_name.lower().replace(' ', '_')}.csv"
        stats.to_csv(csv_file)
        print(f"Saved CSV statistics to: {csv_file}")


if __name__ == "__main__":
    main()
