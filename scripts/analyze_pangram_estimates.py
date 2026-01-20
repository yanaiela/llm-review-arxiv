#!/usr/bin/env python3
"""
Analyze paper estimates from multiple domains comparing pangram vs all papers.
Extracts data from LaTeX tables and creates visualization.
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple


def parse_latex_table(filepath: str) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], List[int]]:
    """
    Parse LaTeX table to extract review and regular paper estimates.
    
    Returns:
        Tuple of (data dict, list of years)
        data structure: {subset: {year: {review_count, regular_count}}}
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = {
        'All': {},
        'Pangram': {}
    }
    
    all_years = []
    
    # Find All section: from \textbf{All} to \textbf{LLM (Alpha)}
    all_start = content.find(r'\textbf{All}')
    all_end = content.find(r'\textbf{LLM (Alpha)}')
    all_section = content[all_start:all_end]
    
    # Find Pangram section: from \textbf{LLM (Pangram)} to \bottomrule
    pangram_start = content.find(r'\textbf{LLM (Pangram)}')
    pangram_end = content.find(r'\bottomrule')
    pangram_section = content[pangram_start:pangram_end]
    
    # Parse All section
    for line in all_section.split('\n'):
        if r'\quad' in line and '&' in line:
            year_match = re.search(r'\\quad\s+(\d{4})', line)
            if year_match:
                year = int(year_match.group(1))
                all_years.append(year)
                
                parts = line.split('&')
                if len(parts) >= 5:
                    try:
                        review_count = int(parts[1].strip().replace(',', ''))
                        regular_count = int(parts[3].strip().replace(',', ''))
                        data['All'][year] = {'review': review_count, 'regular': regular_count}
                    except ValueError:
                        pass
    
    # Parse Pangram section
    for line in pangram_section.split('\n'):
        if r'\quad' in line and '&' in line:
            year_match = re.search(r'\\quad\s+(\d{4})', line)
            if year_match:
                year = int(year_match.group(1))
                
                parts = line.split('&')
                if len(parts) >= 5:
                    try:
                        review_count = int(parts[1].strip().replace(',', ''))
                        regular_count = int(parts[3].strip().replace(',', ''))
                        data['Pangram'][year] = {'review': review_count, 'regular': regular_count}
                    except ValueError:
                        pass
    
    return data, sorted(set(all_years))


def load_all_domains(data_dir: str = "/home/nlp/lazary/workspace/ai-review-slop/data/tables") -> Dict[str, Dict]:
    """Load data from all 4 domain tables."""
    domains = ['cs', 'math', 'physics', 'stat']
    all_data = {}
    
    for domain in domains:
        filepath = f"{data_dir}/paper_number_estimates_{domain}_combined.tex"
        data, years = parse_latex_table(filepath)
        all_data[domain] = {'data': data, 'years': years}
    
    return all_data


def create_visualization(all_data: Dict[str, Dict], domains: List[str] = None, output_path: str = None):
    """
    Create figure showing pangram review vs regular papers.
    
    Args:
        all_data: Dictionary containing data for all domains
        domains: List of domain keys to plot. If None, defaults to all 4 domains
        output_path: Optional custom output path
    """
    if domains is None:
        domains = ['cs', 'math', 'physics', 'stat']
    
    domain_mapping = {
        'cs': 'Computer Science',
        'math': 'Mathematics',
        'physics': 'Physics',
        'stat': 'Statistics'
    }
    
    num_domains = len(domains)
    
    # Determine subplot layout
    if num_domains == 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    elif num_domains == 2:
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = [axes]
    
    # Use seaborn color palette
    colors = sns.color_palette("husl", 2)
    
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        domain_data = all_data[domain]['data']
        years = sorted(all_data[domain]['years'])
        
        # Get pangram and all counts
        pangram_review = [domain_data['Pangram'][year]['review'] for year in years]
        pangram_regular = [domain_data['Pangram'][year]['regular'] for year in years]
        all_review = [domain_data['All'][year]['review'] for year in years]
        all_regular = [domain_data['All'][year]['regular'] for year in years]
        
        spacing_factor = 0.55  # tighten horizontal spacing between year groups
        x = np.arange(len(years)) * spacing_factor
        width = 0.25
        
        # Plot bars
        bars1 = ax.bar(x - width/2, pangram_review, width, label='Review', color=colors[0])
        bars2 = ax.bar(x + width/2, pangram_regular, width, label='Non-Review', color=colors[1])
        
        # Add value labels above bars
        for i, (year, pang_rev, pang_reg, all_rev, all_reg) in enumerate(zip(years, pangram_review, pangram_regular, all_review, all_regular)):
            # Review bar label
            pct_review = (pang_rev / all_rev * 100) if all_rev > 0 else 0
            ax.text(x[i] - width/2, pang_rev, f'{pang_rev}\n({pct_review:.1f}%)', 
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
            
            # Regular bar label
            pct_regular = (pang_reg / all_reg * 100) if all_reg > 0 else 0
            ax.text(x[i] + width/2, pang_reg, f'{pang_reg}\n({pct_regular:.1f}%)', 
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        # Set y-axis range with some padding
        y_max = max(max(pangram_review), max(pangram_regular))
        ax.set_ylim(0, y_max * 1.15)
        
        ax.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax.set_ylabel('LLM-Generated Papers Estimates', fontsize=16, fontweight='bold')
        ax.set_title(f'{domain_mapping[domain]}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=16)
    
    plt.tight_layout()
    
    if output_path is None:
        # Generate default filename based on domains
        if domains == ['cs', 'math', 'physics', 'stat']:
            filename = 'pangram_estimates.png'
        elif domains == ['cs', 'physics']:
            filename = 'pangram_estimates_cs_physics.png'
        elif domains == ['math', 'stat']:
            filename = 'pangram_estimates_math_stats.png'
        else:
            filename = f"pangram_estimates_{'_'.join(domains)}.png"
        
        output_path = f"/home/nlp/lazary/workspace/ai-review-slop/output/figures/{filename}"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {pdf_path}")
    
    plt.close()


def print_summary_table(all_data: Dict[str, Dict]):
    """Print summary of extracted pangram data."""
    print("\n" + "="*80)
    print("SUMMARY: Pangram Paper Estimates")
    print("="*80)
    
    for domain in ['cs', 'math', 'physics', 'stat']:
        print(f"\n{domain.upper()}:")
        print("-" * 80)
        domain_data = all_data[domain]['data']
        years = sorted(all_data[domain]['years'])
        
        for year in years:
            pang_rev = domain_data['Pangram'][year]['review']
            pang_reg = domain_data['Pangram'][year]['regular']
            
            print(f"\n  Year {year}:")
            print(f"    Review:  {pang_rev:>8,}")
            print(f"    Regular: {pang_reg:>8,}")
            
            # Calculate review percentage
            pang_pct = (pang_rev / (pang_rev + pang_reg)) * 100 if (pang_rev + pang_reg) > 0 else 0
            print(f"    Review %%: {pang_pct:.1f}%")


if __name__ == "__main__":
    # Load data from all domains
    print("Loading data from all 4 domains...")
    all_data = load_all_domains()
    
    # Print summary
    print_summary_table(all_data)
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    
    # All 4 domains
    create_visualization(all_data)
    
    # CS and Physics
    create_visualization(all_data, domains=['cs', 'physics'])
    
    # Math and Stats
    create_visualization(all_data, domains=['math', 'stat'])
    
    print("\nDone!")
