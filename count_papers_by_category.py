#!/usr/bin/env python3
"""
Script to count the total number of papers in major categories and cs-* subcategories
under the data/processed folder.
"""

import os
import csv
from pathlib import Path
from collections import defaultdict

def count_papers_in_category(category_path):
    """Count the number of papers in a category by reading the CSV metadata file."""
    metadata_file = os.path.join(category_path, 'paper_metadata.csv')
    
    if not os.path.exists(metadata_file):
        return 0
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            # Count rows excluding header
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            return sum(1 for _ in reader)
    except Exception as e:
        print(f"Error reading {metadata_file}: {e}")
        return 0

def count_papers_by_year(category_path):
    """Count papers by year in a category. Returns dict {year: count}."""
    metadata_file = os.path.join(category_path, 'paper_metadata.csv')
    year_counts = defaultdict(int)
    
    if not os.path.exists(metadata_file):
        return year_counts
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                year_month = row.get('year_month', '')
                if year_month:
                    year = year_month[:4]  # Extract YYYY from YYYY-MM
                    year_counts[year] += 1
    except Exception as e:
        print(f"Error reading {metadata_file}: {e}")
    
    return year_counts

def generate_latex_main_categories_table(main_categories, base_path, category_names, years):
    """Generate LaTeX table for main categories with year columns."""
    # Build column spec
    col_spec = 'l' + 'r' * len(years) + 'r'  # Category + years + Total
    
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        r"        \toprule",
    ]
    
    # Header row
    header = r"        \textbf{Category}"
    for year in years:
        header += f" & {year}"
    header += r" & \textbf{n.} \\"
    lines.append(header)
    
    lines.append(r"        \midrule")
    
    # Data rows
    year_totals = defaultdict(int)
    main_total = 0
    
    for category in main_categories:
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            year_counts = count_papers_by_year(category_path)
            display_name = category_names.get(category, category)
            
            row = f"        {display_name}"
            category_total = 0
            for year in years:
                count = year_counts.get(year, 0)
                year_totals[year] += count
                category_total += count
                row += f" & {count:,}"
            
            main_total += category_total
            row += f" & {category_total:,} \\\\"
            lines.append(row)
    
    # Total row
    lines.append(r"        \midrule")
    total_row = r"        Total"
    for year in years:
        total_row += f" & {year_totals[year]:,}"
    total_row += f" & {main_total:,} \\\\"
    lines.append(total_row)
    
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Number of papers in main categories}",
        r"    \label{tab:main_categories}",
        r"\end{table}",
    ])
    
    latex_code = "\n".join(lines)
    return latex_code, main_total

def generate_latex_cs_subcategories_table(cs_subcategories, base_path, subcategory_names, years):
    """Generate LaTeX table for CS subcategories with year columns."""
    # Build column spec
    col_spec = 'l' + 'r' * len(years) + 'r'  # Subcategory + years + Total
    
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        r"        \toprule",
    ]
    
    # Header row
    header = r"        \textbf{Subcategory}"
    for year in years:
        header += f" & {year}"
    header += r" & \textbf{n.} \\"
    lines.append(header)
    
    lines.append(r"        \midrule")
    
    # Data rows
    year_totals = defaultdict(int)
    cs_total = 0
    
    for subcategory in cs_subcategories:
        category_path = os.path.join(base_path, subcategory)
        if os.path.exists(category_path):
            year_counts = count_papers_by_year(category_path)
            display_name = subcategory_names.get(subcategory, subcategory)
            
            row = f"        {display_name}"
            subcategory_total = 0
            for year in years:
                count = year_counts.get(year, 0)
                year_totals[year] += count
                subcategory_total += count
                row += f" & {count:,}"
            
            cs_total += subcategory_total
            row += f" & {subcategory_total:,} \\\\"
            lines.append(row)
    
    # Total row
    lines.append(r"        \midrule")
    total_row = r"        Total"
    for year in years:
        total_row += f" & {year_totals[year]:,}"
    total_row += f" & {cs_total:,} \\\\"
    lines.append(total_row)
    
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Number of papers in CS subcategories}",
        r"    \label{tab:cs_subcategories}",
        r"\end{table}",
    ])
    
    latex_code = "\n".join(lines)
    return latex_code, cs_total

def main():
    base_path = '/home/nlp/lazary/workspace/ai-review-slop/data/processed'
    
    # Main categories
    main_categories = ['cs', 'math', 'physics', 'stat']
    category_names = {
        'cs': 'Computer Science',
        'math': 'Mathematics',
        'physics': 'Physics',
        'stat': 'Statistics'
    }
    
    # CS subcategories
    cs_subcategories = [
        'cs-AI', 'cs-CL', 'cs-CR', 'cs-CV', 'cs-CY', 
        'cs-HC', 'cs-IR', 'cs-LG', 'cs-RO', 'cs-SE'
    ]
    
    subcategory_names = {
        'cs-AI': 'Artificial Intelligence',
        'cs-CL': 'Computation and Language',
        'cs-CR': 'Cryptography and Security',
        'cs-CV': 'Computer Vision and Pattern Recognition',
        'cs-CY': 'Computers and Society',
        'cs-HC': 'Human-Computer Interaction',
        'cs-IR': 'Information Retrieval',
        'cs-LG': 'Machine Learning',
        'cs-RO': 'Robotics',
        'cs-SE': 'Software Engineering'
    }
    
    print("=" * 60)
    print("PAPER COUNT BY CATEGORY")
    print("=" * 60)
    
    # Count papers in main categories
    print("\nMAIN CATEGORIES:")
    print("-" * 60)
    main_total = 0
    for category in main_categories:
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            count = count_papers_in_category(category_path)
            main_total += count
            display_name = category_names.get(category, category)
            print(f"{display_name:20s}: {count:6d} papers")
        else:
            print(f"{category:20s}: [NOT FOUND]")
    
    print("-" * 60)
    print(f"{'TOTAL (Main)':20s}: {main_total:6d} papers")
    
    # Count papers in CS subcategories
    print("\nCS SUBCATEGORIES:")
    print("-" * 60)
    cs_total = 0
    for subcategory in cs_subcategories:
        category_path = os.path.join(base_path, subcategory)
        if os.path.exists(category_path):
            count = count_papers_in_category(category_path)
            cs_total += count
            display_name = subcategory_names.get(subcategory, subcategory)
            print(f"{display_name:40s}: {count:6d} papers")
        else:
            print(f"{subcategory:40s}: [NOT FOUND]")
    
    print("-" * 60)
    print(f"{'TOTAL (CS Sub)':40s}: {cs_total:6d} papers")
    
    # Grand total
    grand_total = main_total + cs_total
    print("\n" + "=" * 60)
    print(f"GRAND TOTAL (All categories): {grand_total:6d} papers")
    print("=" * 60)
    
    # Generate LaTeX tables
    print("\n" + "=" * 60)
    print("LATEX TABLES")
    print("=" * 60)
    
    # Define years for the table (you can adjust this range as needed)
    years = ['2020','2021','2022','2023','2024', '2025']
    
    latex_main, _ = generate_latex_main_categories_table(main_categories, base_path, category_names, years)
    latex_cs, _ = generate_latex_cs_subcategories_table(cs_subcategories, base_path, subcategory_names, years)
    
    print("\n--- Main Categories Table ---")
    print(latex_main)
    
    print("\n--- CS Subcategories Table ---")
    print(latex_cs)
    
    # Save LaTeX tables to files
    output_dir = '/home/nlp/lazary/workspace/ai-review-slop/data/tables'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'main_categories_table.tex'), 'w') as f:
        f.write(latex_main)
    
    with open(os.path.join(output_dir, 'cs_subcategories_table.tex'), 'w') as f:
        f.write(latex_cs)
    
    print(f"\nLaTeX tables saved to {output_dir}/")

if __name__ == '__main__':
    main()
