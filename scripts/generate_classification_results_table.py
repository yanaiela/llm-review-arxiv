#!/usr/bin/env python3
"""
Generate a LaTeX table from classification results JSON files.

This script reads the classification metrics JSON files and creates a compact
LaTeX table showing precision, recall, and F1 scores for each model.
"""

import json
from pathlib import Path


def generate_classification_table():
    """Generate the classification results table from JSON files."""

    # Define the directory containing the results
    results_dir = Path(__file__).parent.parent / "data" / "results" / "classification_results"

    # Model names to display (ordered as desired)
    model_files = [
        ("llama3.3_test_metrics.json", "Llama 3.3"),
        ("gemma2_test_metrics.json", "Gemma 2"),
        ("gpt-oss_test_metrics.json", "GPT-OSS"),
        ("gpt4o-mini_test_metrics.json", "GPT-4o-mini"),
    ]

    # Collect results
    results = []
    for filename, display_name in model_files:
        file_path = results_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append({
                    'model': display_name,
                    'precision': data['precision'],
                    'recall': data['recall'],
                    'f1': data['f1_score']
                })
        else:
            print(f"Warning: {filename} not found")

    # Generate LaTeX table rows
    rows = []
    for r in results:
        # Format as percentages with 1 decimal place
        precision = f"{r['precision']*100:.1f}"
        recall = f"{r['recall']*100:.1f}"
        f1 = f"{r['f1']*100:.1f}"
        rows.append(f"{r['model']} & {precision} & {recall} & {f1} \\\\")

    # Build the complete LaTeX table
    latex_table = f"""\\begin{{table}}[t]
\\centering
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}}}
\\caption{{Classification performance of different models on the test set. All metrics are reported as percentages.}}
\\label{{tab:classification-results}}
\\end{{table}}
"""

    # Save to file
    output_dir = Path(__file__).parent.parent / "data" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "classification_results_table.tex"

    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"✓ Generated classification results table: {output_file}")
    print(f"\nResults:")
    for r in results:
        print(f"  {r['model']:15s} P: {r['precision']*100:5.1f}%  R: {r['recall']*100:5.1f}%  F1: {r['f1']*100:5.1f}%")


if __name__ == "__main__":
    generate_classification_table()
