#!/usr/bin/env python3
"""
Analyze pangram detection results for pre-LLM era papers (2020-2022).
Computes the percentage of papers classified as AI-generated for both reviews and regular papers.
"""

import json
from pathlib import Path
from collections import defaultdict


def main():
    # Load the pangram detection results
    results_file = Path(__file__).parent.parent / "data/results/cs/pangram_detection_results.json"

    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)

    print(f"Total papers loaded: {len(data)}")

    # Initialize counters
    stats = defaultdict(lambda: {'total': 0, 'ai_generated': 0})

    # Process each paper
    for paper in data:
        year = paper.get('year')
        paper_type = paper.get('paper_type')
        period = paper.get('period')

        # Filter for pre-LLM era (2020-2022)
        if year not in [2020, 2021, 2022]:
            continue

        # Verify it's marked as pre_llm period
        if period != 'pre_llm':
            continue

        # Get the prediction
        pangram_pred = paper.get('pangram_prediction', {})
        prediction = pangram_pred.get('prediction', '')

        # Count totals
        stats[paper_type]['total'] += 1

        # Count AI-generated papers (prediction is "Likely AI" or "Very Likely AI")
        if prediction in ['Likely AI', 'Very Likely AI']:
            stats[paper_type]['ai_generated'] += 1

    # Print results
    print("\n" + "="*70)
    print("PRE-LLM ERA (2020-2022) AI DETECTION ANALYSIS")
    print("="*70)

    for paper_type in sorted(stats.keys()):
        total = stats[paper_type]['total']
        ai_count = stats[paper_type]['ai_generated']

        if total > 0:
            percentage = (ai_count / total) * 100
        else:
            percentage = 0.0

        print(f"\n{paper_type.upper()} PAPERS:")
        print(f"  Total papers: {total:,}")
        print(f"  Classified as AI-generated: {ai_count:,}")
        print(f"  Percentage: {percentage:.2f}%")

    # Overall statistics
    total_all = sum(s['total'] for s in stats.values())
    ai_all = sum(s['ai_generated'] for s in stats.values())

    if total_all > 0:
        percentage_all = (ai_all / total_all) * 100
    else:
        percentage_all = 0.0

    print(f"\nOVERALL (ALL PAPER TYPES):")
    print(f"  Total papers: {total_all:,}")
    print(f"  Classified as AI-generated: {ai_all:,}")
    print(f"  Percentage: {percentage_all:.2f}%")
    print("="*70)

    # Detailed breakdown by year
    print("\n" + "="*70)
    print("BREAKDOWN BY YEAR")
    print("="*70)

    year_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'ai_generated': 0}))

    for paper in data:
        year = paper.get('year')
        paper_type = paper.get('paper_type')

        if year not in [2020, 2021, 2022]:
            continue

        pangram_pred = paper.get('pangram_prediction', {})
        prediction = pangram_pred.get('prediction', '')

        year_stats[year][paper_type]['total'] += 1

        if prediction in ['Likely AI', 'Very Likely AI']:
            year_stats[year][paper_type]['ai_generated'] += 1

    for year in sorted(year_stats.keys()):
        print(f"\n{year}:")
        for paper_type in sorted(year_stats[year].keys()):
            total = year_stats[year][paper_type]['total']
            ai_count = year_stats[year][paper_type]['ai_generated']
            percentage = (ai_count / total) * 100 if total > 0 else 0.0

            print(f"  {paper_type}: {ai_count}/{total} ({percentage:.2f}%)")

    print("="*70)


if __name__ == "__main__":
    main()
