"""
Script to compute all high-level categories (e.g., cs, math, physics, stat)
from the arXiv Kaggle dataset.

This script reads the arXiv metadata and extracts unique high-level categories
from the category strings.
"""

import json
import logging
from pathlib import Path
from collections import Counter
from typing import Set, Dict
import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_high_level_category(category: str, use_cs_subcategories: bool = False) -> str:
    """
    Extract the high-level category from an arXiv category code.

    Args:
        category: Full category code (e.g., 'cs.AI', 'math.OC', 'physics.gen-ph')
        use_cs_subcategories: If True, preserve CS subcategories (e.g., 'cs.AI', 'cs.LG')
                             instead of collapsing to 'cs'. Also preserves 'stat.ML'.

    Returns:
        High-level category (e.g., 'cs', 'math', 'physics') or
        CS subcategory if use_cs_subcategories=True (e.g., 'cs.AI', 'cs.LG')
    """
    # Handle CS subcategories separately if requested
    if use_cs_subcategories:
        # Preserve CS subcategories and stat.ML
        if category.startswith('cs.'):
            return category

    # Split on '.' or '-' to get the prefix
    # Examples: cs.AI -> cs, physics.gen-ph -> physics
    if '.' in category:
        return category.split('.')[0]
    elif '-' in category:
        return category.split('-')[0]
    else:
        # Some old arXiv categories don't have dots
        return category


def compute_arxiv_categories(
    metadata_file: Path = None,
    max_papers: int = None,
    post_llm_year: int = 2023,
    return_yearly: bool = False,
    use_cs_subcategories: bool = False
) -> Dict[str, int]:
    """
    Compute all high-level categories from the arXiv Kaggle dataset.

    Args:
        metadata_file: Path to the arXiv metadata file. If None, downloads from Kaggle.
        max_papers: Maximum number of papers to process (for testing). None = all papers.
        post_llm_year: Year to start counting for post-LLM period (default: 2023).
        return_yearly: If True, return yearly breakdown of categories.
        use_cs_subcategories: If True, preserve CS subcategories (cs.AI, cs.LG, etc.)
                             instead of collapsing to 'cs'.

    Returns:
        Dictionary mapping high-level category to count
    """
    # Download or get path to the dataset if not provided
    if metadata_file is None:
        logger.info("Locating arXiv dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("Cornell-University/arxiv")
        logger.info(f"Dataset path: {dataset_path}")
        metadata_file = Path(dataset_path) / 'arxiv-metadata-oai-snapshot.json'

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    logger.info(f"Reading metadata from {metadata_file}")

    # Track high-level categories
    high_level_categories = Counter()
    category_year_counts = {}
    total_papers = 0
    papers_with_categories = 0

    # Read the JSON file line by line
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_papers and line_num > max_papers:
                break

            if line_num % 100000 == 0:
                logger.info(f"Processed {line_num:,} papers")

            try:
                paper = json.loads(line)
                total_papers += 1

                # Get categories
                categories_str = paper.get('categories', '')
                if not categories_str:
                    continue

                papers_with_categories += 1

                # Parse categories (space-separated)
                if isinstance(categories_str, str):
                    categories = categories_str.split()
                else:
                    continue

                # Get year
                year = None
                if 'update_date' in paper:
                    try:
                        year = int(paper['update_date'][:4])
                    except Exception:
                        year = None
                elif 'submitted_date' in paper:
                    try:
                        year = int(paper['submitted_date'][:4])
                    except Exception:
                        year = None

                # Extract high-level categories
                for category in categories:
                    high_level = extract_high_level_category(category, use_cs_subcategories)
                    if high_level:
                        # For post-LLM filter
                        if year is not None and year >= post_llm_year:
                            high_level_categories[high_level] += 1
                        # For yearly breakdown
                        if return_yearly and year is not None:
                            if high_level not in category_year_counts:
                                category_year_counts[high_level] = {}
                            if year not in category_year_counts[high_level]:
                                category_year_counts[high_level][year] = 0
                            category_year_counts[high_level][year] += 1

            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue

    logger.info(f"\nProcessing complete!")
    logger.info(f"Total papers processed: {total_papers:,}")
    logger.info(f"Papers with categories: {papers_with_categories:,}")
    logger.info(f"Papers without categories: {total_papers - papers_with_categories:,}")

    if return_yearly:
        return dict(high_level_categories), category_year_counts
    else:
        return dict(high_level_categories)


def print_category_report(categories: Dict[str, int]):
    """
    Print a formatted report of high-level categories.

    Args:
        categories: Dictionary mapping category to count
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"HIGH-LEVEL ARXIV CATEGORIES")
    logger.info(f"{'='*60}")
    logger.info(f"Total unique high-level categories: {len(categories)}")
    logger.info(f"\nCategories sorted by frequency:")
    logger.info(f"{'-'*60}")
    logger.info(f"{'Category':<20} {'Count':>15} {'Percentage':>10}")
    logger.info(f"{'-'*60}")

    # Sort by count (descending)
    total_count = sum(categories.values())
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)

    for category, count in sorted_categories:
        percentage = (count / total_count) * 100
        logger.info(f"{category:<20} {count:>15,} {percentage:>9.2f}%")

    logger.info(f"{'-'*60}")
    logger.info(f"{'TOTAL':<20} {total_count:>15,} {100.0:>9.2f}%")
    logger.info(f"{'='*60}")

    # Print categories alphabetically for easy reference
    logger.info(f"\nAll categories (alphabetical):")
    logger.info(f"{', '.join(sorted(categories.keys()))}")


def save_categories_to_file(categories: Dict[str, int], output_file: Path):
    """
    Save categories to a JSON file.

    Args:
        categories: Dictionary mapping category to count
        output_file: Path to output JSON file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create output with metadata
    output_data = {
        'total_categories': len(categories),
        'total_papers': sum(categories.values()),
        'categories': dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved category data to {output_file}")


def main():
    """Main function to compute and display arXiv categories."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute high-level categories from arXiv Kaggle dataset'
    )
    parser.add_argument(
        '--metadata-file',
        type=Path,
        help='Path to arXiv metadata JSON file (if not specified, downloads from Kaggle)'
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        help='Maximum number of papers to process (for testing)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/arxiv_categories.json'),
        help='Output file for category data (default: data/arxiv_categories.json)'
    )
    parser.add_argument(
        '--use-cs-subcategories',
        action='store_true',
        help='Preserve CS subcategories (cs.AI, cs.LG, etc.) instead of collapsing to cs'
    )

    args = parser.parse_args()

    # Compute categories
    categories = compute_arxiv_categories(
        metadata_file=args.metadata_file,
        max_papers=args.max_papers,
        use_cs_subcategories=args.use_cs_subcategories
    )

    # Print report
    print_category_report(categories)

    # Save to file
    save_categories_to_file(categories, args.output)


if __name__ == "__main__":
    main()
