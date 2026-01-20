"""
Script to analyze CS subcategories from ArXiv sampled papers.
Reads the CS papers from the kaggle folder and computes subcategory counts.
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import sys


def extract_cs_subcategories(categories_str):
    """
    Extract CS subcategories from a categories string.

    Args:
        categories_str: Space-separated string of categories (e.g., "cs.AI cs.LG")

    Returns:
        List of CS subcategories (e.g., ["cs.AI", "cs.LG"])
    """
    if pd.isna(categories_str) or not categories_str:
        return []

    # Split by space and filter for CS categories
    categories = categories_str.split()
    cs_categories = [cat for cat in categories if cat.startswith('cs.')]

    return cs_categories


def count_cs_subcategories(csv_path):
    """
    Count CS subcategories from the ArXiv sampled papers.

    Args:
        csv_path: Path to the CSV file with sampled papers

    Returns:
        Counter object with subcategory counts
    """
    # Read the CSV file
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total papers: {len(df):,}")

    # Extract all CS subcategories
    all_cs_subcategories = []
    for categories in df['categories']:
        cs_cats = extract_cs_subcategories(categories)
        all_cs_subcategories.extend(cs_cats)

    # Count occurrences
    subcategory_counts = Counter(all_cs_subcategories)

    return subcategory_counts


def print_subcategory_stats(counts):
    """
    Print statistics about CS subcategories.

    Args:
        counts: Counter object with subcategory counts
    """
    print("\n" + "="*60)
    print("CS SUBCATEGORY STATISTICS")
    print("="*60)

    # Sort by count (descending) then by name
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    total_occurrences = sum(counts.values())
    print(f"\nTotal subcategory occurrences: {total_occurrences:,}")
    print(f"Unique subcategories: {len(counts)}")
    print("\n" + "-"*60)
    print(f"{'Subcategory':<20} {'Count':>10} {'Percentage':>10}")
    print("-"*60)

    for subcategory, count in sorted_counts:
        percentage = (count / total_occurrences) * 100
        print(f"{subcategory:<20} {count:>10,} {percentage:>9.2f}%")

    print("-"*60)
    print(f"{'TOTAL':<20} {total_occurrences:>10,} {100.0:>9.2f}%")
    print("="*60)


def save_subcategory_stats(counts, output_path):
    """
    Save subcategory statistics to a CSV file.

    Args:
        counts: Counter object with subcategory counts
        output_path: Path to save the CSV file
    """
    # Sort by count (descending)
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    # Create DataFrame
    total = sum(counts.values())
    data = []
    for subcategory, count in sorted_counts:
        percentage = (count / total) * 100
        data.append({
            'subcategory': subcategory,
            'count': count,
            'percentage': f"{percentage:.2f}"
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\nStatistics saved to {output_path}")


def main():
    # Path to the CS sampled papers
    kaggle_dir = Path(__file__).parent.parent / 'data' / 'kaggle'
    csv_path = kaggle_dir / 'arxiv_sampled_papers_cs.csv'

    # Check if file exists
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        print("\nAvailable files in kaggle directory:")
        for f in kaggle_dir.glob('*.csv'):
            print(f"  - {f.name}")
        sys.exit(1)

    # Count subcategories
    counts = count_cs_subcategories(csv_path)

    # Print statistics
    print_subcategory_stats(counts)

    # Save to CSV
    output_path = kaggle_dir / 'cs_subcategory_counts.csv'
    save_subcategory_stats(counts, output_path)


if __name__ == "__main__":
    main()
