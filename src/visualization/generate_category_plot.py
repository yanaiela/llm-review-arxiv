#!/usr/bin/env python3
"""
Generate adjusted alpha by category plot.
"""

import yaml
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization.create_plots import ResultsVisualizer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate adjusted alpha by category plots')
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

    # Load config
    config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create visualizer with appropriate detection method
    detection_method = 'pangram' if args.pangram else 'alpha'
    visualizer = ResultsVisualizer(config, detection_method=detection_method)

    # Generate the plots
    category_type = "CS subcategories" if args.subcategories else "high-level categories"
    detection_label = "Pangram" if args.pangram else "adjusted"
    print(f"Generating {detection_label} alpha by category plot using {category_type}...")
    visualizer.plot_adjusted_alpha_by_category(use_cs_subcategories=args.subcategories)

    suffix = '_cs_subcategories' if args.subcategories else ''
    suffix += '_pangram' if args.pangram else ''
    print(f"Plot saved to {visualizer.fig_dir / f'adjusted_alpha_by_category{suffix}.pdf'}")

    print(f"\nGenerating {detection_label} alpha by category and year plot using {category_type}...")
    visualizer.plot_adjusted_alpha_by_category_and_year(use_cs_subcategories=args.subcategories)
    print(f"Plot saved to {visualizer.fig_dir / f'adjusted_alpha_by_category_and_year{suffix}.pdf'}")

if __name__ == '__main__':
    main()
