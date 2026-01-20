"""
Script to generate the tab:yearly-alpha LaTeX table from the paper.
This table shows alpha estimates (proportion of AI-generated text) by year for review and regular papers.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def load_statistical_analysis(category: str, use_adjusted: bool = True) -> Dict:
    """
    Load statistical analysis results for a given category.

    Args:
        category: The arXiv category (e.g., 'cs', 'cs.AI', 'math', 'stat', 'physics')
        use_adjusted: Whether to use adjusted (baseline-corrected) results

    Returns:
        Dictionary with statistical analysis results
    """
    cat = category.replace('.', '-')

    if use_adjusted:
        results_path = Path(f"data/results/{cat}/adjusted/statistical_analysis.json")
    else:
        results_path = Path(f"data/results/{cat}/statistical_analysis.json")

    if not results_path.exists():
        raise FileNotFoundError(f"Statistical analysis file not found: {results_path}")

    with open(results_path, 'r') as f:
        data = json.load(f)

    return data


def extract_yearly_alpha_data(data: Dict, use_adjusted: bool = True) -> pd.DataFrame:
    """
    Extract yearly alpha data from statistical analysis results.

    Args:
        data: Statistical analysis dictionary
        use_adjusted: Whether to use adjusted (baseline-corrected) alpha values

    Returns:
        DataFrame with yearly alpha data
    """
    yearly_alphas = data['yearly_trends']['alpha_comparison_by_year']

    rows = []
    for year_str, alpha_data in yearly_alphas.items():
        year = int(year_str)

        if use_adjusted:
            # Use adjusted values if available
            review_alpha = alpha_data.get('review_alpha_adjusted', alpha_data['review_alpha'])
            regular_alpha = alpha_data.get('regular_alpha_adjusted', alpha_data['regular_alpha'])
            review_ci_lower = alpha_data.get('review_ci_lower_adjusted', alpha_data['review_ci_lower'])
            review_ci_upper = alpha_data.get('review_ci_upper_adjusted', alpha_data['review_ci_upper'])
            regular_ci_lower = alpha_data.get('regular_ci_lower_adjusted', alpha_data['regular_ci_lower'])
            regular_ci_upper = alpha_data.get('regular_ci_upper_adjusted', alpha_data['regular_ci_upper'])
        else:
            review_alpha = alpha_data['review_alpha']
            regular_alpha = alpha_data['regular_alpha']
            review_ci_lower = alpha_data['review_ci_lower']
            review_ci_upper = alpha_data['review_ci_upper']
            regular_ci_lower = alpha_data['regular_ci_lower']
            regular_ci_upper = alpha_data['regular_ci_upper']

        rows.append({
            'year': year,
            'review_alpha': review_alpha,
            'review_ci_lower': review_ci_lower,
            'review_ci_upper': review_ci_upper,
            'regular_alpha': regular_alpha,
            'regular_ci_lower': regular_ci_lower,
            'regular_ci_upper': regular_ci_upper,
            'difference': review_alpha - regular_alpha,
        })

    df = pd.DataFrame(rows).sort_values('year')
    return df


def generate_latex_table(
    df: pd.DataFrame,
    category_name: str,
    use_adjusted: bool = True,
    show_ci: bool = True
) -> str:
    """
    Generate LaTeX table code for yearly alpha estimates.

    Args:
        df: DataFrame with yearly alpha data
        category_name: Name of the category for the caption
        use_adjusted: Whether the data uses adjusted values
        show_ci: Whether to show confidence intervals

    Returns:
        LaTeX table code as string
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\resizebox{\\columnwidth}{!}{%")

    if show_ci:
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Year} & \\textbf{Review} $\\alpha$ & \\textbf{95\\% CI} & "
                    "\\textbf{Regular} $\\alpha$ & \\textbf{95\\% CI} \\\\")
    else:
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Year} & \\textbf{Review} $\\alpha$ & "
                    "\\textbf{Regular} $\\alpha$ & \\textbf{Difference} \\\\")

    latex.append("\\midrule")

    # Add data rows
    for _, row in df.iterrows():
        year = int(row['year'])
        review_alpha = row['review_alpha']
        regular_alpha = row['regular_alpha']

        if show_ci:
            review_ci = f"[{row['review_ci_lower']:.3f}, {row['review_ci_upper']:.3f}]"
            regular_ci = f"[{row['regular_ci_lower']:.3f}, {row['regular_ci_upper']:.3f}]"
            latex.append(
                f"{year} & {review_alpha:.3f} & {review_ci} & "
                f"{regular_alpha:.3f} & {regular_ci} \\\\"
            )
        else:
            difference = row['difference']
            latex.append(
                f"{year} & {review_alpha:.3f} & {regular_alpha:.3f} & "
                f"{difference:+.3f} \\\\"
            )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")

    # Add caption
    adjusted_text = "baseline-corrected " if use_adjusted else ""
    latex.append(
        f"\\caption{{Yearly {adjusted_text}alpha estimates ($\\alpha$) for {category_name} papers. "
        f"Alpha represents the estimated proportion of AI-generated text in abstracts.}}"
    )
    latex.append("\\label{tab:yearly-alpha}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_multi_category_table(
    categories: List[str],
    category_names: Optional[Dict[str, str]] = None,
    use_adjusted: bool = True
) -> str:
    """
    Generate a combined LaTeX table showing alpha values for multiple categories.

    Args:
        categories: List of category codes (e.g., ['cs', 'math', 'stat', 'physics'])
        category_names: Optional mapping of category codes to display names
        use_adjusted: Whether to use adjusted (baseline-corrected) alpha values

    Returns:
        LaTeX table code as string
    """
    if category_names is None:
        category_names = {
            'cs': 'Computer Science',
            'cs.AI': 'AI',
            'cs.CL': 'Computation and Language',
            'cs.CV': 'Computer Vision',
            'cs.LG': 'Machine Learning',
            'math': 'Mathematics',
            'stat': 'Statistics',
            'physics': 'Physics'
        }

    # Collect data for all categories
    all_data = {}
    years = set()

    for category in categories:
        try:
            data = load_statistical_analysis(category, use_adjusted)
            df = extract_yearly_alpha_data(data, use_adjusted)
            all_data[category] = df
            years.update(df['year'].tolist())
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    if not all_data:
        raise ValueError("No data found for any category")

    years = sorted(years)

    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\resizebox{\\columnwidth}{!}{%")

    # Dynamic column specification based on number of categories
    num_cols = len(all_data) * 2 + 1  # year + (review + regular) per category
    col_spec = "l" + "cc" * len(all_data)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Header row 1: Category names
    header1 = "\\textbf{Year}"
    for category in categories:
        if category in all_data:
            cat_name = category_names.get(category, category)
            header1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{cat_name}}}}}"
    header1 += " \\\\"
    latex.append(header1)

    # Header row 2: Review/Regular
    header2 = ""
    for i, category in enumerate(categories):
        if category in all_data:
            if i == 0:
                header2 += "\\cmidrule(lr){2-3}"
            else:
                header2 += f"\\cmidrule(lr){{{i*2+2}-{i*2+3}}}"
    latex.append(header2)

    header3 = ""
    for category in categories:
        if category in all_data:
            header3 += " & \\textbf{Review} & \\textbf{Regular}"
    header3 += " \\\\"
    latex.append(header3)

    latex.append("\\midrule")

    # Data rows
    for year in years:
        row = f"{year}"
        for category in categories:
            if category in all_data:
                df = all_data[category]
                year_data = df[df['year'] == year]
                if not year_data.empty:
                    review_alpha = year_data.iloc[0]['review_alpha']
                    regular_alpha = year_data.iloc[0]['regular_alpha']
                    row += f" & {review_alpha:.3f} & {regular_alpha:.3f}"
                else:
                    row += " & -- & --"
        row += " \\\\"
        latex.append(row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")

    # Caption
    adjusted_text = "baseline-corrected " if use_adjusted else ""
    latex.append(
        f"\\caption{{Yearly {adjusted_text}alpha estimates by category and paper type. "
        f"Alpha represents the estimated proportion of AI-generated text in abstracts.}}"
    )
    latex.append("\\label{tab:yearly-alpha}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    """Main function to generate yearly alpha tables."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate yearly alpha estimates table for the paper'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Single category to analyze (e.g., cs, math, stat, physics, cs.AI)'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        default=['cs', 'math', 'stat', 'physics'],
        help='Multiple categories to include in table (default: cs math stat physics)'
    )
    parser.add_argument(
        '--use-adjusted',
        action='store_true',
        default=True,
        help='Use baseline-corrected alpha estimates (default: True)'
    )
    parser.add_argument(
        '--no-adjusted',
        action='store_true',
        default=False,
        help='Use raw (non-adjusted) alpha estimates'
    )
    parser.add_argument(
        '--show-ci',
        action='store_true',
        default=False,
        help='Show confidence intervals in single-category table'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/tables',
        help='Output directory for tables (default: data/tables)'
    )

    args = parser.parse_args()

    # Determine whether to use adjusted values
    use_adjusted = args.use_adjusted and not args.no_adjusted

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("GENERATING YEARLY ALPHA TABLE")
    print("=" * 80)
    print(f"Using {'adjusted (baseline-corrected)' if use_adjusted else 'raw'} alpha estimates")

    if args.category:
        # Single category table
        print(f"\nGenerating table for category: {args.category}")

        try:
            data = load_statistical_analysis(args.category, use_adjusted)
            df = extract_yearly_alpha_data(data, use_adjusted)

            print(f"\nYearly Alpha Estimates for {args.category}:")
            print(df.to_string(index=False))

            # Generate LaTeX
            category_names = {
                'cs': 'Computer Science',
                'math': 'Mathematics',
                'stat': 'Statistics',
                'physics': 'Physics',
                'cs.AI': 'Artificial Intelligence',
                'cs.CL': 'Computation and Language',
                'cs.CV': 'Computer Vision',
                'cs.LG': 'Machine Learning',
            }
            category_name = category_names.get(args.category, args.category)

            latex_table = generate_latex_table(df, category_name, use_adjusted, args.show_ci)

            # Save table
            adjusted_suffix = "_adjusted" if use_adjusted else "_raw"
            table_file = output_dir / f"yearly_alpha_{args.category.replace('.', '-')}{adjusted_suffix}.tex"
            with open(table_file, 'w') as f:
                f.write(latex_table)

            print(f"\nSaved LaTeX table to: {table_file}")
            print("\n" + "=" * 80)
            print("LATEX OUTPUT:")
            print("=" * 80)
            print(latex_table)

        except Exception as e:
            print(f"Error processing category {args.category}: {e}")
            return

    else:
        # Multi-category table
        print(f"\nGenerating table for categories: {', '.join(args.categories)}")

        try:
            latex_table = generate_multi_category_table(args.categories, use_adjusted=use_adjusted)

            # Save table
            adjusted_suffix = "_adjusted" if use_adjusted else "_raw"
            table_file = output_dir / f"yearly_alpha_multi{adjusted_suffix}.tex"
            with open(table_file, 'w') as f:
                f.write(latex_table)

            print(f"\nSaved LaTeX table to: {table_file}")
            print("\n" + "=" * 80)
            print("LATEX OUTPUT:")
            print("=" * 80)
            print(latex_table)

        except Exception as e:
            print(f"Error generating multi-category table: {e}")
            return

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
