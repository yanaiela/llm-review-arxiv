"""
Script to plot estimated counts of AI-generated papers per category for post-LLM years.
Uses alpha estimates from statistical analysis and total paper counts from arXiv.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import logging
import argparse

from src.data_collection.compute_arxiv_categories import compute_arxiv_categories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a white background for all plots to stay consistent across visualization scripts
plt.style.use("default")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.edgecolor'] = 'white'


# Mapping from arXiv category codes to human-readable names
CATEGORY_LABELS: Dict[str, str] = {
    # High-level categories
    'cs': 'Computer Science',
    'math': 'Mathematics',
    'physics': 'Physics',
    'stat': 'Statistics',

    # CS subcategories
    'cs.AI': 'Artificial Intelligence',
    'cs.CL': 'Computation and Language',
    'cs.CR': 'Cryptography and Security',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.LG': 'Machine Learning',
    'cs.RO': 'Robotics',
    'cs.SE': 'Software Engineering',
}


def get_category_label(category_code: str) -> str:
    """Return a human-readable label for a given arXiv category code."""
    return CATEGORY_LABELS.get(category_code, category_code)


def load_alpha_estimates(category: str, use_pangram: bool = False) -> Dict:
    """
    Load alpha estimates from statistical analysis results.

    Args:
        category: Category name (cs, math, physics, stat)
        use_pangram: If True, load pangram results. If False, load adjusted results.

    Returns:
        Dictionary with alpha estimates by year and paper type
    """
    cat = category.replace('.', '-')

    if use_pangram:
        results_path = Path(f"data/results/{cat}/pangram_statistical_analysis.json")
    else:
        results_path = Path(f"data/results/{cat}/adjusted/statistical_analysis.json")

    if not results_path.exists():
        raise FileNotFoundError(f"Statistical analysis file not found: {results_path}")

    with open(results_path, 'r') as f:
        data = json.load(f)

    return data


def get_review_percentages(category: str) -> Dict[int, float]:
    """
    Calculate review paper percentages by year from paper_classifications.csv.

    Args:
        category: Category name (cs, math, physics, stat)

    Returns:
        Dictionary mapping year -> review_percentage (0-1)
    """
    cat = category.replace('.', '-')
    csv_path = Path(f"data/processed/{cat}/paper_classifications.csv")

    if not csv_path.exists():
        logger.error(f"Classification file not found for {category}")
        raise FileNotFoundError(f"Classification file not found: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # We need to load paper metadata to get years
    metadata_path = Path(f"data/processed/{cat}/paper_metadata.json")
    if not metadata_path.exists():
        logger.error(f"Metadata file not found for {cat}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create a mapping of arxiv_id to year
    id_to_year = {}
    for paper in metadata:
        arxiv_id = paper.get('id')
        update_date = paper.get('update_date') or paper.get('submitted_date')
        if arxiv_id and update_date:
            try:
                year = int(update_date[:4])
                id_to_year[arxiv_id] = year
            except Exception:
                pass

    # Count review vs regular by year
    year_counts = {}
    for _, row in df.iterrows():
        arxiv_id = row['arxiv_id']
        paper_type = row['predicted_type']

        if arxiv_id in id_to_year:
            year = id_to_year[arxiv_id]
            if year not in year_counts:
                year_counts[year] = {'review': 0, 'regular': 0}

            if paper_type == 'review':
                year_counts[year]['review'] += 1
            else:
                year_counts[year]['regular'] += 1

    # Calculate percentages
    review_percentages = {}
    for year, counts in year_counts.items():
        total = counts['review'] + counts['regular']
        if total > 0:
            review_percentages[year] = counts['review'] / total

    return review_percentages


def get_total_paper_counts_from_arxiv(
    category: str,
    post_llm_year: int = 2023,
    category_year_counts=None
) -> Dict[int, Dict[str, int]]:
    """
    Get total paper counts per year and type from the full arXiv dataset.
    Uses compute_arxiv_categories function and review percentages from classifications.

    Args:
        category: Category name (cs, math, physics, stat)
        metadata_file: Path to arXiv metadata file. If None, downloads from Kaggle.
        post_llm_year: Year when LLM era started

    Returns:
        Dictionary mapping year -> {'review': count, 'regular': count}
    """
    

    if category not in category_year_counts:
        raise ValueError(f"Category {category} not found in arXiv data")

    year_counts = category_year_counts[category]

    # Get review percentages from classifications
    review_percentages = get_review_percentages(category)

    # Split total counts into review/regular based on percentages
    result = {}
    for year, total_count in year_counts.items():
        if year < post_llm_year:
            continue

        # Get review percentage for this year (default to 8% if not available)
        review_pct = review_percentages.get(year, 0.08)

        review_count = int(total_count * review_pct)
        regular_count = total_count - review_count

        result[year] = {
            'review': review_count,
            'regular': regular_count
        }

        logger.info(f"{category} {year}: total={total_count}, review={review_count} ({review_pct:.1%}), regular={regular_count}")

    return result


def compute_ai_generated_estimates(
    category: str,
    post_llm_year: int = 2023,
    category_year_counts=None,
    use_pangram: bool = False
) -> Tuple[Dict, Dict]:
    """
    Compute estimated counts of AI-generated papers based on alpha and total counts from full arXiv.

    Args:
        category: Category name (cs, math, physics, stat)
        post_llm_year: Year when LLM era started
        category_year_counts: Dictionary of category year counts from compute_arxiv_categories
        use_pangram: If True, use pangram results. If False, use adjusted results.

    Returns:
        Tuple of (review_estimates, regular_estimates) where each is a dict mapping year -> count
    """
    data = load_alpha_estimates(category, use_pangram)

    total_counts = get_total_paper_counts_from_arxiv(category, post_llm_year, category_year_counts)

    review_estimates = {}
    regular_estimates = {}

    if use_pangram:
        # Pangram data has a different structure: ai_rates_by_year_type
        ai_rates = data['yearly_trends']['ai_rates_by_year_type']

        # Group by year and type
        for entry in ai_rates:
            year = entry['year']
            paper_type = entry['paper_type']
            ai_rate = entry['ai_rate']

            if year not in total_counts:
                logger.warning(f"No total counts found for {category} in {year}")
                continue

            # Get total counts from full arXiv dataset
            if paper_type == 'review':
                review_total = total_counts[year].get('review', 0)
                review_estimates[year] = ai_rate * review_total
                logger.info(f"{category} {year}: AI-gen review={review_estimates[year]:.0f} (rate={ai_rate:.3f})")
            else:  # regular
                regular_total = total_counts[year].get('regular', 0)
                regular_estimates[year] = ai_rate * regular_total
                logger.info(f"{category} {year}: AI-gen regular={regular_estimates[year]:.0f} (rate={ai_rate:.3f})")
    else:
        # Adjusted data has alpha_comparison_by_year structure
        alpha_by_year = data['yearly_trends']['alpha_comparison_by_year']

        for year_str, alpha_data in alpha_by_year.items():
            year = int(year_str)

            if year not in total_counts:
                logger.warning(f"No total counts found for {category} in {year}")
                continue

            # Get alpha estimates
            review_alpha = alpha_data['review_alpha']
            regular_alpha = alpha_data['regular_alpha']

            # Get total counts from full arXiv dataset
            review_total = total_counts[year].get('review', 0)
            regular_total = total_counts[year].get('regular', 0)

            # Compute estimated AI-generated counts
            review_estimates[year] = review_alpha * review_total
            regular_estimates[year] = regular_alpha * regular_total

            logger.info(f"{category} {year}: AI-gen review={review_estimates[year]:.0f} (α={review_alpha:.3f}), "
                       f"AI-gen regular={regular_estimates[year]:.0f} (α={regular_alpha:.3f})")

    return review_estimates, regular_estimates


def get_target_categories(use_cs_subcategories: bool = False) -> List[str]:
    """
    Get the list of categories to analyze.

    Args:
        use_cs_subcategories: If True, use CS subcategories. If False, use high-level categories.

    Returns:
        List of category names
    """
    if use_cs_subcategories:
        return ['cs.AI', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.CY', 'cs.HC', 'cs.IR', 'cs.LG', 'cs.RO', 'cs.SE']
    else:
        return ['cs', 'math', 'physics', 'stat']


def plot_ai_generated_by_category_and_type(post_llm_year=2023, category_year_counts=None, use_cs_subcategories=False, use_pangram=False):
    """
    Plot estimated AI-generated papers by category, with separate plots for review and regular papers.

    Args:
        post_llm_year: Year when LLM era started
        category_year_counts: Dictionary of category year counts from compute_arxiv_categories
        use_cs_subcategories: If True, use CS subcategories. If False, use high-level categories (default).
        use_pangram: If True, use pangram results. If False, use adjusted results.
    """
    target_categories = get_target_categories(use_cs_subcategories)

    # Collect data for all categories
    review_data = []
    regular_data = []

    for category in target_categories:
        try:
            review_est, regular_est = compute_ai_generated_estimates(category, post_llm_year, category_year_counts, use_pangram)

            for year, count in review_est.items():
                if year >= post_llm_year:
                    review_data.append({
                        'Category': category,
                        'Year': year,
                        'Count': count
                    })

            for year, count in regular_est.items():
                if year >= post_llm_year:
                    regular_data.append({
                        'Category': category,
                        'Year': year,
                        'Count': count
                    })
        except FileNotFoundError as e:
            logger.warning(f"No data found for category {category}: {e}")
            continue

    # Create DataFrames
    review_df = pd.DataFrame(review_data)
    regular_df = pd.DataFrame(regular_data)

    # Add display labels
    if not review_df.empty:
        review_df['Display'] = review_df['Category'].apply(get_category_label)
    if not regular_df.empty:
        regular_df['Display'] = regular_df['Category'].apply(get_category_label)

    # Determine file suffix based on subcategories and pangram flags
    detection_suffix = '_pangram' if use_pangram else ''
    category_suffix = '_cs_subcategories' if use_cs_subcategories else ''

    # Plot 1: Review papers
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=review_df, x='Year', y='Count', hue='Display', marker='o', linewidth=2.5)
    title_method = ' (Pangram)' if use_pangram else ''
    plt.title(f'Estimated AI-Generated Review Papers by Category ({post_llm_year}+){title_method}', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Estimated Count', fontsize=12)
    plt.legend(title='Category', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'data/figures/ai_generated_review_papers_by_category{category_suffix}{detection_suffix}.png', dpi=300)
    plt.show()

    # Plot 2: Regular papers
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=regular_df, x='Year', y='Count', hue='Display', marker='o', linewidth=2.5)
    plt.title(f'Estimated AI-Generated Regular Papers by Category ({post_llm_year}+){title_method}', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Estimated Count', fontsize=12)
    plt.legend(title='Category', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'data/figures/ai_generated_regular_papers_by_category{category_suffix}{detection_suffix}.png', dpi=300)
    plt.show()


def plot_ai_generated_by_category_aggregated(post_llm_year=2023, category_year_counts=None, use_cs_subcategories=False, use_pangram=False):
    """
    Plot total estimated AI-generated papers aggregated across all post-LLM years, by category.
    Shows separate bars for review and regular papers.

    Args:
        post_llm_year: Year when LLM era started
        category_year_counts: Dictionary of category year counts from compute_arxiv_categories
        use_cs_subcategories: If True, use CS subcategories. If False, use high-level categories (default).
        use_pangram: If True, use pangram results. If False, use adjusted results.
    """
    target_categories = get_target_categories(use_cs_subcategories)

    # Collect aggregated data
    data = []

    for category in target_categories:
        try:
            review_est, regular_est = compute_ai_generated_estimates(category, post_llm_year, category_year_counts, use_pangram)

            # Sum across all post-LLM years
            review_total = sum(count for year, count in review_est.items() if year >= post_llm_year)
            regular_total = sum(count for year, count in regular_est.items() if year >= post_llm_year)

            data.append({
                'Category': category,
                'Type': 'Review',
                'Count': review_total
            })
            data.append({
                'Category': category,
                'Type': 'Regular',
                'Count': regular_total
            })
        except FileNotFoundError as e:
            logger.warning(f"No data found for category {category}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(data)
    if not df.empty:
        df['Display'] = df['Category'].apply(get_category_label)

    # Determine file suffix and figure size based on subcategories and pangram flags
    detection_suffix = '_pangram' if use_pangram else ''
    category_suffix = '_cs_subcategories' if use_cs_subcategories else ''
    fig_width = 14 if use_cs_subcategories else 10

    # Plot
    plt.figure(figsize=(fig_width, 6))
    sns.barplot(data=df, x='Display', y='Count', hue='Type', palette='Set2')
    title_method = ' (Pangram)' if use_pangram else ''
    plt.title(f'Total Estimated AI-Generated Papers by Category ({post_llm_year}+){title_method}', fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Estimated Count', fontsize=12)
    plt.legend(title='Paper Type', fontsize=10)
    # Rotate labels if using subcategories
    if use_cs_subcategories:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'data/figures/ai_generated_papers_aggregated{category_suffix}{detection_suffix}.png', dpi=300)
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate AI-generated paper count plots by category'
    )
    parser.add_argument(
        '--subcategories',
        action='store_true',
        default=False,
        help='Use CS subcategories (cs.LG, cs.CV, etc.) instead of high-level categories (cs, math, stat, physics)'
    )
    parser.add_argument(
        '--pangram',
        action='store_true',
        default=False,
        help='Use Pangram detection results instead of adjusted results'
    )
    args = parser.parse_args()

    metadata_file = None  # Will download from Kaggle if needed
    post_llm_year = 2023

    # Create figures directory if it doesn't exist
    Path('data/figures').mkdir(parents=True, exist_ok=True)

    # Get total counts by year using existing function
    category_type = "CS subcategories" if args.subcategories else "high-level categories"
    detection_method = "Pangram" if args.pangram else "adjusted"
    logger.info(f"Using {category_type} for analysis with {detection_method} detection method")

    _, category_year_counts = compute_arxiv_categories(
        metadata_file=metadata_file,
        post_llm_year=post_llm_year,
        return_yearly=True,
        use_cs_subcategories=args.subcategories
    )

    # Generate plots
    plot_ai_generated_by_category_and_type(post_llm_year, category_year_counts, args.subcategories, args.pangram)
    plot_ai_generated_by_category_aggregated(post_llm_year, category_year_counts, args.subcategories, args.pangram)


if __name__ == "__main__":
    main()
