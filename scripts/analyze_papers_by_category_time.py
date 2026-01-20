#!/usr/bin/env python3
"""
Script to analyze arXiv papers by category and time period.
Reads all arxiv_sampled_papers_cs-* JSON files and shows the number of papers
per category per month/year.
"""

import json
import glob
import os
from collections import defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_papers_from_files(data_dir="data/kaggle"):
    """Load all arxiv_sampled_papers_cs-*.json files."""
    pattern = os.path.join(data_dir, "arxiv_sampled_papers_cs-*.json")
    files = glob.glob(pattern)

    all_papers = []
    for file_path in sorted(files):
        category = os.path.basename(file_path).replace("arxiv_sampled_papers_", "").replace(".json", "")
        print(f"Loading {category}...")

        with open(file_path, 'r') as f:
            papers = json.load(f)
            for paper in papers:
                paper['main_category'] = category
                all_papers.append(paper)

    print(f"\nTotal papers loaded: {len(all_papers)}")
    return all_papers


def analyze_by_month(papers):
    """Count papers by category and month."""
    counts = defaultdict(lambda: defaultdict(int))

    for paper in papers:
        category = paper['main_category']
        year_month = paper.get('year_month')

        if year_month:
            counts[category][year_month] += 1

    return counts


def analyze_by_year(papers):
    """Count papers by category and year."""
    counts = defaultdict(lambda: defaultdict(int))

    for paper in papers:
        category = paper['main_category']
        year_month = paper.get('year_month')

        if year_month:
            year = year_month.split('-')[0]
            counts[category][year] += 1

    return counts


def create_dataframe_monthly(counts):
    """Convert monthly counts to a pandas DataFrame."""
    data = []
    for category, months in counts.items():
        for month, count in months.items():
            data.append({
                'category': category,
                'year_month': month,
                'count': count
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['year_month', 'category'])
    return df


def create_dataframe_yearly(counts):
    """Convert yearly counts to a pandas DataFrame."""
    data = []
    for category, years in counts.items():
        for year, count in years.items():
            data.append({
                'category': category,
                'year': year,
                'count': count
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['year', 'category'])
    return df


def print_summary(df_monthly, df_yearly):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY BY YEAR")
    print("="*80)
    print(df_yearly.to_string(index=False))

    print("\n" + "="*80)
    print("TOTAL PAPERS PER CATEGORY")
    print("="*80)
    category_totals = df_yearly.groupby('category')['count'].sum().sort_values(ascending=False)
    for category, total in category_totals.items():
        print(f"{category}: {total:,}")

    print("\n" + "="*80)
    print("TOTAL PAPERS PER YEAR")
    print("="*80)
    year_totals = df_yearly.groupby('year')['count'].sum().sort_values()
    for year, total in year_totals.items():
        print(f"{year}: {total:,}")

    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN (First 20 rows)")
    print("="*80)
    print(df_monthly.head(20).to_string(index=False))


def plot_trends(df_monthly, df_yearly, output_dir="output/figures"):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)

    # Plot 1: Papers per category per year (stacked bar chart)
    fig, ax = plt.subplots(figsize=(14, 8))
    df_pivot = df_yearly.pivot(index='year', columns='category', values='count').fillna(0)
    df_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Papers', fontsize=12)
    ax.set_title('Papers per Category per Year (Stacked)', fontsize=14, fontweight='bold')
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'papers_per_category_year_stacked.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/papers_per_category_year_stacked.png")
    plt.close()

    # Plot 2: Papers per category per year (grouped bar chart)
    fig, ax = plt.subplots(figsize=(14, 8))
    df_pivot.plot(kind='bar', ax=ax, colormap='tab20')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Papers', fontsize=12)
    ax.set_title('Papers per Category per Year (Grouped)', fontsize=14, fontweight='bold')
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'papers_per_category_year_grouped.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/papers_per_category_year_grouped.png")
    plt.close()

    # Plot 3: Time series per category (monthly)
    fig, ax = plt.subplots(figsize=(16, 8))
    for category in df_monthly['category'].unique():
        category_data = df_monthly[df_monthly['category'] == category].sort_values('year_month')
        ax.plot(category_data['year_month'], category_data['count'], marker='o', label=category, linewidth=2)

    ax.set_xlabel('Year-Month', fontsize=12)
    ax.set_ylabel('Number of Papers', fontsize=12)
    ax.set_title('Papers per Category Over Time (Monthly)', fontsize=14, fontweight='bold')
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    # Show every nth label to avoid crowding
    n = max(1, len(ax.get_xticklabels()) // 20)
    for i, label in enumerate(ax.get_xticklabels()):
        if i % n != 0:
            label.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'papers_per_category_monthly_timeseries.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/papers_per_category_monthly_timeseries.png")
    plt.close()

    # Plot 4: Heatmap of papers by category and year
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_pivot.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Papers'})
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_title('Heatmap: Papers per Category per Year', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'papers_heatmap_category_year.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/papers_heatmap_category_year.png")
    plt.close()


def save_csv_reports(df_monthly, df_yearly, output_dir="output/reports"):
    """Save CSV reports."""
    os.makedirs(output_dir, exist_ok=True)

    monthly_path = os.path.join(output_dir, 'papers_by_category_month.csv')
    df_monthly.to_csv(monthly_path, index=False)
    print(f"\nSaved: {monthly_path}")

    yearly_path = os.path.join(output_dir, 'papers_by_category_year.csv')
    df_yearly.to_csv(yearly_path, index=False)
    print(f"Saved: {yearly_path}")


def main():
    """Main function."""
    print("ArXiv Papers Analysis by Category and Time")
    print("=" * 80)

    # Load all papers
    papers = load_papers_from_files()

    # Analyze by month and year
    monthly_counts = analyze_by_month(papers)
    yearly_counts = analyze_by_year(papers)

    # Create DataFrames
    df_monthly = create_dataframe_monthly(monthly_counts)
    df_yearly = create_dataframe_yearly(yearly_counts)

    # Print summary
    print_summary(df_monthly, df_yearly)

    # Save CSV reports
    # save_csv_reports(df_monthly, df_yearly)

    # Create visualizations
    # plot_trends(df_monthly, df_yearly)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
