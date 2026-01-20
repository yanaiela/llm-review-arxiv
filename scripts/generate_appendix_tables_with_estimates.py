"""
Enhanced script to generate LaTeX tables matching the paper's appendix format.
Creates tables with both sample counts and estimated totals extrapolated to full arXiv.

The extrapolation is based on the ratio between our sample size (6000 papers/year for most categories)
and the estimated total arXiv submissions for each category/year.

ArXiv submission statistics can be found at: https://arxiv.org/stats/monthly_submissions
These estimates are approximate and based on typical arXiv growth patterns.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_metadata(category: str) -> pd.DataFrame:
    """Load paper metadata for a given category."""
    data_dir = Path(__file__).parent.parent / 'data'
    metadata_path = data_dir / 'processed' / category / 'paper_metadata.csv'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    df['submission_date'] = pd.to_datetime(df['submission_date'], errors='coerce')
    df['year'] = df['submission_date'].dt.year

    return df


def load_classifications(category: str) -> pd.DataFrame:
    """Load paper classifications for a given category."""
    data_dir = Path(__file__).parent.parent / 'data'
    classifications_path = data_dir / 'processed' / category / 'paper_classifications.csv'

    if not classifications_path.exists():
        raise FileNotFoundError(f"Classifications file not found: {classifications_path}")

    return pd.read_csv(classifications_path)


def merge_data(metadata: pd.DataFrame, classifications: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata with classifications."""
    return metadata.merge(classifications, on='arxiv_id', how='left')


def compute_arxiv_estimates_from_data() -> Dict[str, Dict[int, int]]:
    """
    Compute actual total arXiv submissions for each category and year
    by counting papers in the full arXiv metadata file from Kaggle.

    This follows the same procedure as the data collection process in
    src/data_collection/compute_arxiv_categories.py

    Returns:
        Dictionary mapping category -> year -> total count
    """
    import kagglehub

    # Download or get path to the full arXiv dataset
    print("Locating full arXiv dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("Cornell-University/arxiv")
    print(f"Dataset path: {dataset_path}")

    metadata_file = Path(dataset_path) / 'arxiv-metadata-oai-snapshot.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    print(f"Reading full arXiv metadata from {metadata_file}")
    print("This may take several minutes...")

    # Track category-year counts
    category_year_counts = {
        'cs': {},
        'math': {},
        'stat': {},
        'physics': {}
    }

    total_papers = 0
    processed_papers = 0

    # Read the JSON file line by line
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"Processed {line_num:,} papers...")

            try:
                paper = json.loads(line)
                total_papers += 1

                # Get categories
                categories_str = paper.get('categories', '')
                if not categories_str:
                    continue

                # Parse categories (space-separated)
                if isinstance(categories_str, str):
                    categories = categories_str.split()
                else:
                    continue

                # Get year from update_date
                year = None
                if 'update_date' in paper:
                    try:
                        year = int(paper['update_date'][:4])
                    except Exception:
                        year = None

                if year is None:
                    continue

                # Only count papers from 2023-2025
                if not (2023 <= year <= 2025):
                    continue

                # Extract high-level categories
                high_level_categories = set()
                for category in categories:
                    # Get high-level category (e.g., cs.AI -> cs)
                    if '.' in category:
                        high_level = category.split('.')[0]
                    elif '-' in category:
                        high_level = category.split('-')[0]
                    else:
                        high_level = category

                    # Only track our target categories
                    if high_level in category_year_counts:
                        high_level_categories.add(high_level)

                # Only count papers with a single high-level category
                if len(high_level_categories) == 1:
                    high_level = list(high_level_categories)[0]
                    if year not in category_year_counts[high_level]:
                        category_year_counts[high_level][year] = 0
                    category_year_counts[high_level][year] += 1
                    processed_papers += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue

    print(f"\nProcessing complete!")
    print(f"Total papers read: {total_papers:,}")
    print(f"Papers counted (2023-2025): {processed_papers:,}")

    return category_year_counts


def load_alpha_estimates(category: str) -> Dict[int, Dict[str, float]]:
    """
    Load alpha estimates (AI-generation rates) per year and paper type.

    Args:
        category: Category key ('cs', 'math', 'stat', 'physics')

    Returns:
        Dictionary mapping year -> {'review': alpha, 'regular': alpha}
    """
    data_dir = Path(__file__).parent.parent / 'data'
    stats_path = data_dir / 'results' / category / 'adjusted' / 'statistical_analysis.json'

    if not stats_path.exists():
        raise FileNotFoundError(f"Statistical analysis file not found: {stats_path}")

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    alpha_by_year = {}
    yearly_data = stats.get('yearly_trends', {}).get('alpha_comparison_by_year', {})

    for year_str, year_data in yearly_data.items():
        year = int(year_str)
        alpha_by_year[year] = {
            'review': year_data.get('review_alpha', 0.0),
            'regular': year_data.get('regular_alpha', 0.0)
        }

    return alpha_by_year


def load_pangram_rates(category: str) -> Dict[int, Dict[str, float]]:
    """
    Load pangram detection rates (AI-generation rates) per year and paper type.

    Args:
        category: Category key ('cs', 'math', 'stat', 'physics')

    Returns:
        Dictionary mapping year -> {'review': rate, 'regular': rate}
    """
    data_dir = Path(__file__).parent.parent / 'data'
    stats_path = data_dir / 'results' / category / 'pangram_statistical_analysis.json'

    if not stats_path.exists():
        raise FileNotFoundError(f"Pangram statistical analysis file not found: {stats_path}")

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    rates_by_year = {}
    yearly_data = stats.get('yearly_trends', {}).get('ai_rates_by_year_type', [])

    for entry in yearly_data:
        year = entry['year']
        paper_type = entry['paper_type']
        ai_rate = entry['ai_rate']

        if year not in rates_by_year:
            rates_by_year[year] = {}

        rates_by_year[year][paper_type] = ai_rate

    return rates_by_year


def get_arxiv_estimate(category: str, year: int, estimates_cache: Dict[str, Dict[int, int]] = None) -> int:
    """
    Get estimated total arXiv submissions for a category and year.

    Args:
        category: Category key ('cs', 'math', 'stat', 'physics')
        year: Year (2023-2025)
        estimates_cache: Precomputed estimates from compute_arxiv_estimates_from_data()

    Returns:
        Estimated total submissions
    """
    # Use cached estimates if provided
    if estimates_cache and category in estimates_cache:
        return estimates_cache[category].get(year, -1)

    # Fallback if not found: assume 8x multiplier (sample is ~12.5% of total)
    return -1


def compute_year_statistics_with_estimates(
    df: pd.DataFrame,
    category: str,
    estimates_cache: Dict[str, Dict[int, int]] = None,
    ai_rate_method: Optional[str] = None,
    ai_rates: Optional[Dict[int, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    Compute statistics by year and paper type with estimated totals.

    Args:
        df: DataFrame with merged metadata and classifications
        category: Category key (e.g., 'cs', 'math')
        estimates_cache: Precomputed estimates from compute_arxiv_estimates_from_data()
        ai_rate_method: Method for computing AI-generated estimates ('alpha', 'pangram', or None)
        ai_rates: AI generation rates per year/type (when ai_rate_method is set)

    Returns:
        DataFrame with year-level statistics including estimates
    """
    # Filter to valid years (2023-2025)
    df_filtered = df[df['year'].between(2023, 2025)].copy()

    # Create paper type column
    df_filtered['is_review'] = df_filtered['predicted_type'] == 'review'

    # Count papers by year and type
    stats_list = []

    for year in sorted(df_filtered['year'].unique()):
        year_df = df_filtered[df_filtered['year'] == year]

        review_sample = (year_df['is_review'] == True).sum()
        regular_sample = (year_df['is_review'] == False).sum()
        sample_total = len(year_df)

        # Get estimated total for this year/category
        arxiv_total = get_arxiv_estimate(category, year, estimates_cache)

        # Calculate scaling factor
        if sample_total > 0:
            scale_factor = arxiv_total / sample_total
        else:
            scale_factor = 1.0

        if ai_rates:
            review_ai_rate = ai_rates[int(year)].get('review', 0.0)
            regular_ai_rate = ai_rates[int(year)].get('regular', 0.0)
        else:
            review_ai_rate = 1
            regular_ai_rate = 1
        
        # Estimate totals (AI-generated or full totals)
        review_est = int(review_sample * scale_factor * review_ai_rate)
        regular_est = int(regular_sample * scale_factor * regular_ai_rate)
        total_est = review_est + regular_est

        # Calculate percentages based on estimated totals
        review_pct = (review_est / total_est * 100) if total_est > 0 else 0
        regular_pct = (regular_est / total_est * 100) if total_est > 0 else 0

        stats_list.append({
            'year': year,
            'Review Sample': review_sample,
            'Non-Review Sample': regular_sample,
            'Review Est. Total': review_est,
            'Non-Review Est. Total': regular_est,
            'Total Est.': total_est,
            'Review %': round(review_pct, 1),
            'Non-Review %': round(regular_pct, 1),
        })

    stats = pd.DataFrame(stats_list)
    stats.set_index('year', inplace=True)

    return stats


def generate_appendix_latex_table(stats: pd.DataFrame, category_name: str, ai_rate_method: Optional[str] = None) -> str:
    """
    Generate LaTeX table in the exact format used in the paper's appendix.

    Args:
        stats: DataFrame with statistics
        category_name: Name of the category for the caption
        ai_rate_method: Method used for AI-generation estimates ('alpha', 'pangram', or None)

    Returns:
        LaTeX table code as string
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\resizebox{\\columnwidth}{!}{%")
    latex.append("\\begin{tabular}{lrrrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Year} & \\multicolumn{2}{c}{\\textbf{Review Papers}} & \\multicolumn{2}{c}{\\textbf{Non-Review Papers}} \\\\")
    latex.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    latex.append(" & \\textbf{n.} & \\textbf{\\%} & \\textbf{n.} & \\textbf{\\%} \\\\")
    latex.append("\\midrule")

    # Add data rows
    total_review_est = 0
    total_regular_est = 0

    for year in sorted(stats.index):
        row = stats.loc[year]

        review_est = int(row['Review Est. Total'])
        regular_est = int(row['Non-Review Est. Total'])
        review_pct = row['Review %']
        regular_pct = row['Non-Review %']

        total_review_est += review_est
        total_regular_est += regular_est

        latex.append(
            f"{int(year)} & {review_est:,} & "
            f"{review_pct:.1f}\\% & "
            f"{regular_est:,} & {regular_pct:.1f}\\% \\\\"
        )

    # Add totals row
    latex.append("\\midrule")
    total_est = total_review_est + total_regular_est
    total_review_pct = (total_review_est / total_est * 100) if total_est > 0 else 0
    total_regular_pct = (total_regular_est / total_est * 100) if total_est > 0 else 0

    latex.append(
        f"\\textbf{{Total}} & \\textbf{{{total_review_est:,}}} & "
        f"\\textbf{{{total_review_pct:.1f}\\%}} & "
        f"\\textbf{{{total_regular_est:,}}} & \\textbf{{{total_regular_pct:.1f}\\%}} \\\\"
    )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")

    # Customize caption based on AI rate method
    if ai_rate_method == 'alpha':
        caption = (
            f"\\caption{{Estimated number of AI-generated {category_name} papers per year by paper type "
            f"(using alpha estimates). Sample shows our dataset; Est. Total shows extrapolated estimates "
            f"for AI-generated papers in all arXiv papers in this category.}}"
        )
    elif ai_rate_method == 'pangram':
        caption = (
            f"\\caption{{Estimated number of AI-generated {category_name} papers per year by paper type "
            f"(using pangram detection). Sample shows our dataset; Est. Total shows extrapolated estimates "
            f"for AI-generated papers in all arXiv papers in this category.}}"
        )
    else:
        caption = (
            f"\\caption{{Number of {category_name} papers per year by paper type. "
            f"Sample shows our dataset; Est. Total shows extrapolated estimates for all arXiv papers in this category.}}"
        )

    latex.append(caption)
    label_suffix = f"-{ai_rate_method}" if ai_rate_method else ""
    latex.append(f"\\label{{tab:papers-{category_name.lower().replace(' ', '-')}-stats{label_suffix}}}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_combined_appendix_latex_table(stats_dict: Dict[str, pd.DataFrame], category_name: str) -> str:
    """
    Generate a combined LaTeX table showing all three estimation methods with years grouped within each method.

    Args:
        stats_dict: Dictionary mapping estimation method ('none', 'alpha', 'pangram') to DataFrame with statistics
        category_name: Name of the category for the caption

    Returns:
        LaTeX table code as string
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\resizebox{\\columnwidth}{!}{%")
    # 5 columns: method/year + review total + review % + non-review total + non-review %
    latex.append("\\begin{tabular}{lrrrr}")
    latex.append("\\toprule")
    
    # Header row
    latex.append("\\textbf{Subset / Year} & \\multicolumn{2}{c}{\\textbf{Review Papers}} & \\multicolumn{2}{c}{\\textbf{Non-Review Papers}} \\\\")
    latex.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    latex.append(" & \\textbf{n.} & \\textbf{\\%} & \\textbf{n.} & \\textbf{\\%} \\\\")
    latex.append("\\midrule")

    # Collect all stats by year
    all_years = set()
    for stats in stats_dict.values():
        all_years.update(stats.index)
    all_years = sorted(all_years)
    
    # Add data rows, grouped by method with year blocks
    all_totals = {method: {'review': 0, 'regular': 0} for method in stats_dict.keys()}
    
    method_names = {
        'none': 'All',
        'alpha': 'LLM (Alpha)',
        'pangram': 'LLM (Pangram)'
    }
    
    for j, method in enumerate(['none', 'alpha', 'pangram']):
        method_total_review = 0
        method_total_regular = 0
        
        if j > 0:
            # Add spacing between method blocks
            latex.append("\\addlinespace")
        
        # Print method header
        latex.append(f"\\textbf{{{method_names[method]}}}")
        latex.append(" & & & & \\\\")
        
        # Print years for this method
        for year in all_years:
            if method in stats_dict and year in stats_dict[method].index:
                row = stats_dict[method].loc[year]
                review_est = int(row['Review Est. Total'])
                regular_est = int(row['Non-Review Est. Total'])
                review_pct = row['Review %']
                regular_pct = row['Non-Review %']
                
                method_total_review += review_est
                method_total_regular += regular_est
                all_totals[method]['review'] += review_est
                all_totals[method]['regular'] += regular_est
                
                latex.append(
                    f"\\quad {int(year)} & {review_est:,} & {review_pct:.1f}\\% & "
                    f"{regular_est:,} & {regular_pct:.1f}\\% \\\\"
                )
            else:
                # Missing data for this method/year
                latex.append(f"\\quad {int(year)} & - & - & - & - \\\\")
        
        # Add total row for this method
        # method_total = method_total_review + method_total_regular
        # method_review_pct = (method_total_review / method_total * 100) if method_total > 0 else 0
        # method_regular_pct = (method_total_regular / method_total * 100) if method_total > 0 else 0
        
        # latex.append(
        #     f"\\quad \\textbf{{Total}} & \\textbf{{{method_total_review:,}}} & "
        #     f"\\textbf{{{method_review_pct:.1f}\\%}} & \\textbf{{{method_total_regular:,}}} & "
        #     f"\\textbf{{{method_regular_pct:.1f}\\%}} \\\\"
        # )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")

    caption = (
        f"\\caption{{Comparison of {category_name} paper estimates by estimation method and year. "
        f"For each subset, we show the estimated review vs. non-review papers for each year (2023-2025). }}"
    )
    latex.append(caption)
    latex.append(f"\\label{{tab:papers-{category_name.lower().replace(' ', '-')}-stats-combined}}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    """Main function to generate all tables with estimates."""
    categories = {
        'cs': 'Computer Science',
        'math': 'Mathematics',
        'stat': 'Statistics',
        'physics': 'Physics'
    }

    output_dir = Path(__file__).parent.parent / 'data' / 'tables'
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("GENERATING COMBINED APPENDIX TABLES WITH ALL ESTIMATION METHODS")
    print("=" * 80)
    print("\nComputing arXiv estimates from actual data...")

    # Compute estimates once from the Kaggle data
    estimates_cache = compute_arxiv_estimates_from_data()

    # Print the computed estimates for verification
    print("\nComputed estimates from data:")
    for cat, year_counts in estimates_cache.items():
        print(f"\n{cat}:")
        for year in sorted(year_counts.keys()):
            print(f"  {year}: {year_counts[year]:,}")

    print("\n" + "=" * 80)

    all_combined_latex = []

    for category_key, category_name in categories.items():
        print(f"\n{category_name}")
        print("=" * 80)

        try:
            # Load data
            metadata = load_metadata(category_key)
            classifications = load_classifications(category_key)

            # Merge data
            df = merge_data(metadata, classifications)

            # Dictionary to store stats for each estimation method
            stats_dict = {}
            
            # Loop through all three options: no estimates, alpha, pangram
            for ai_estimate_method in [None, 'alpha', 'pangram']:
                method_label = ai_estimate_method or 'no estimates'
                print(f"\nProcessing {method_label}...")
                
                # Load AI rates if needed
                ai_rates = None
                if ai_estimate_method == 'alpha':
                    print(f"Loading alpha estimates for {category_key}...")
                    ai_rates = load_alpha_estimates(category_key)
                elif ai_estimate_method == 'pangram':
                    print(f"Loading pangram rates for {category_key}...")
                    ai_rates = load_pangram_rates(category_key)

                # Compute statistics with estimates
                stats = compute_year_statistics_with_estimates(
                    df, category_key, estimates_cache,
                    ai_rate_method=ai_estimate_method,
                    ai_rates=ai_rates
                )

                # Store stats with key for combined table
                key = ai_estimate_method or 'none'
                stats_dict[key] = stats
                
                # Print statistics
                print(f"Statistics for {method_label}:")
                print(stats.to_string())

            # Generate combined LaTeX table
            combined_latex_table = generate_combined_appendix_latex_table(stats_dict, category_name)

            all_combined_latex.append(f"\n\\subsection{{{category_name}}}\n\\label{{app:dataset-stats-{category_key}-combined}}\n\n{combined_latex_table}\n")

            # Save combined table
            combined_table_file = output_dir / f"paper_number_estimates_{category_key}_combined.tex"
            with open(combined_table_file, 'w') as f:
                f.write(combined_latex_table)
            print(f"\nSaved combined table to: {combined_table_file}")

        except Exception as e:
            print(f"Error processing {category_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save all combined tables to a single file for easy inclusion in appendix
    # appendix_file = output_dir / "paper_number_estimates_combined.tex"

    # with open(appendix_file, 'w') as f:
    #     f.write("% Auto-generated appendix tables with all estimation methods\n")
    #     f.write("% Generated by scripts/generate_appendix_tables_with_estimates.py\n\n")
    #     f.write("\\section{Dataset Statistics by Year, Category, and Estimation Method}\n")
    #     f.write("\\label{app:dataset-stats-combined}\n\n")
    #     intro_text = (
    #         "This section provides detailed statistics on the number of papers per year in our dataset, "
    #         "broken down by scientific category and paper type (review vs. non-review). For each category, "
    #         "we show estimates under three different scenarios: (1) total paper counts, (2) AI-generated paper counts "
    #         "using alpha rates, and (3) AI-generated paper counts using pangram detection. For each scenario, "
    #         "we show both our sample counts and estimated totals extrapolated to the full arXiv population. "
    #         "The extrapolations are based on the sampling ratio between our dataset and the total number "
    #         "of arXiv papers in each category per year, multiplied by the AI-generation rate (for methods 2 and 3).\\footnote{Estimated totals are based on arXiv "
    #         "submission statistics available at \\url{https://arxiv.org/stats/monthly_submissions}.}\n\n"
    #     )
    #     f.write(intro_text)
    #     f.write("\n".join(all_combined_latex))

    # print("\n" + "=" * 80)
    # print(f"All combined tables saved to: {appendix_file}")
    # print("=" * 80)
    # print("\nTo use in your LaTeX document, include:")
    # try:
    #     relative_path = appendix_file.relative_to(Path.cwd())
    #     print(f"  \\input{{{relative_path}}}")
    # except ValueError:
    #     print(f"  \\input{{{appendix_file}}}")


if __name__ == "__main__":
    main()
