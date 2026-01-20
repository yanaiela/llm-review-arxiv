#!/usr/bin/env python3
"""
Script to check for empty titles or abstracts in paper metadata CSV file.
"""

import pandas as pd
import sys
from pathlib import Path

def check_empty_fields(csv_path):
    """
    Check how many rows have empty title or abstract fields.

    Args:
        csv_path: Path to the CSV file
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Get total number of rows
    total_rows = len(df)

    # Check for empty titles
    empty_titles = df['title'].isna() | (df['title'].str.strip() == '')
    num_empty_titles = empty_titles.sum()

    # Check for empty abstracts
    empty_abstracts = df['abstract'].isna() | (df['abstract'].str.strip() == '')
    num_empty_abstracts = empty_abstracts.sum()

    # Check for rows with either empty title or abstract
    empty_either = empty_titles | empty_abstracts
    num_empty_either = empty_either.sum()

    # Check for rows with both empty title and abstract
    empty_both = empty_titles & empty_abstracts
    num_empty_both = empty_both.sum()

    # Print results
    print(f"Analysis of: {csv_path}")
    print(f"{'='*60}")
    print(f"Total rows: {total_rows}")
    print(f"\nEmpty fields:")
    print(f"  - Empty titles: {num_empty_titles} ({num_empty_titles/total_rows*100:.2f}%)")
    print(f"  - Empty abstracts: {num_empty_abstracts} ({num_empty_abstracts/total_rows*100:.2f}%)")
    print(f"  - Empty title OR abstract: {num_empty_either} ({num_empty_either/total_rows*100:.2f}%)")
    print(f"  - Empty title AND abstract: {num_empty_both} ({num_empty_both/total_rows*100:.2f}%)")

    # Show some examples if there are any empty fields
    if num_empty_either > 0:
        print(f"\n{'='*60}")
        print("Sample rows with empty fields (first 5):")
        print(df[empty_either][['title', 'abstract']].head())

if __name__ == "__main__":
    csv_path = "data/processed/cs/paper_metadata.csv"
    check_empty_fields(csv_path)
