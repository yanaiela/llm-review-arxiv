#!/usr/bin/env python3
"""
Script to count papers from the processed directory per month/year per category.
Generates a table showing the distribution of papers across categories and time periods.
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def get_processed_categories(processed_dir='data/processed'):
    """Get list of category directories in processed folder."""
    processed_path = Path(processed_dir)
    categories = [d.name for d in processed_path.iterdir()
                  if d.is_dir() and not d.name.startswith('.')]
    return sorted(categories)

def load_category_metadata(category, processed_dir='data/processed'):
    """Load paper metadata for a given category."""
    metadata_file = Path(processed_dir) / category / 'paper_metadata.csv'
    if not metadata_file.exists():
        print(f"Warning: {metadata_file} not found")
        return None

    try:
        df = pd.read_csv(metadata_file)
        return df
    except Exception as e:
        print(f"Error loading {metadata_file}: {e}")
        return None

def count_papers_by_period(processed_dir='data/processed'):
    """Count papers by month/year and category."""
    categories = get_processed_categories(processed_dir)

    # Dictionary to store counts: {year_month: {category: count}}
    period_category_counts = defaultdict(lambda: defaultdict(int))
    category_totals = defaultdict(int)
    period_totals = defaultdict(int)

    for category in categories:
        df = load_category_metadata(category, processed_dir)
        if df is None or df.empty:
            continue

        # Check if year_month column exists
        if 'year_month' not in df.columns:
            print(f"Warning: 'year_month' column not found in {category}")
            continue

        # Count papers per year_month
        counts = df['year_month'].value_counts()

        for period, count in counts.items():
            period_category_counts[period][category] = count
            period_totals[period] += count

        category_totals[category] = len(df)

    return period_category_counts, category_totals, period_totals, categories

def print_table(period_category_counts, category_totals, period_totals, categories):
    """Print a formatted table of paper counts."""

    # Sort periods chronologically
    periods = sorted(period_category_counts.keys())

    # Calculate column widths
    period_width = max(len("Period"), max(len(str(p)) for p in periods) if periods else 0)
    category_widths = {cat: max(len(cat), 6) for cat in categories}
    total_width = 7

    # Print header
    header = f"{'Period':<{period_width}}"
    for cat in categories:
        header += f" | {cat:>{category_widths[cat]}}"
    header += f" | {'Total':>{total_width}}"
    print(header)
    print("-" * len(header))

    # Print data rows
    for period in periods:
        row = f"{period:<{period_width}}"
        for cat in categories:
            count = period_category_counts[period].get(cat, 0)
            row += f" | {count:>{category_widths[cat]}}"
        row += f" | {period_totals[period]:>{total_width}}"
        print(row)

    # Print separator
    print("-" * len(header))

    # Print totals
    total_row = f"{'TOTAL':<{period_width}}"
    for cat in categories:
        total_row += f" | {category_totals[cat]:>{category_widths[cat]}}"
    grand_total = sum(category_totals.values())
    total_row += f" | {grand_total:>{total_width}}"
    print(total_row)

def print_summary_stats(period_category_counts, category_totals, period_totals, categories):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal number of papers: {sum(category_totals.values())}")
    print(f"Number of categories: {len(categories)}")
    print(f"Number of time periods: {len(period_category_counts)}")

    if period_category_counts:
        periods = sorted(period_category_counts.keys())
        print(f"Date range: {periods[0]} to {periods[-1]}")

    print("\nPapers per category:")
    for cat in sorted(categories, key=lambda x: category_totals[x], reverse=True):
        count = category_totals[cat]
        pct = (count / sum(category_totals.values()) * 100) if sum(category_totals.values()) > 0 else 0
        print(f"  {cat:15s}: {count:6d} ({pct:5.2f}%)")

def main():
    """Main function."""
    print("Counting papers by period and category...")
    print()

    period_category_counts, category_totals, period_totals, categories = count_papers_by_period()

    if not period_category_counts:
        print("No data found in processed directory.")
        return

    print_table(period_category_counts, category_totals, period_totals, categories)
    print_summary_stats(period_category_counts, category_totals, period_totals, categories)

if __name__ == "__main__":
    main()
