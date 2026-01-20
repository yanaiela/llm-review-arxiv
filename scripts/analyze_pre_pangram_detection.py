#!/usr/bin/env python3
"""
Script to analyze pre_pangram_detection_results.csv files.
For each category, calculate the percentage of documents detected as AI
(anything not labeled as 'Unlikely AI').
"""

import pandas as pd
import json
import os
from pathlib import Path


def extract_prediction_label(pangram_prediction):
    """Extract the prediction label from the pangram_prediction string."""
    try:
        # The pangram_prediction column contains a string representation of a dict
        pred_dict = eval(pangram_prediction)
        return pred_dict.get('prediction', None)
    except:
        return None


def analyze_category(csv_path, category_name):
    """Analyze a single category CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        # Extract prediction labels
        df['prediction_label'] = df['pangram_prediction'].apply(extract_prediction_label)
        
        # Count total documents
        total_docs = len(df)
        
        # Count documents NOT labeled as "Unlikely AI"
        ai_detected = df[df['prediction_label'] != 'Unlikely AI'].shape[0]
        
        # Calculate percentage
        ai_percentage = (ai_detected / total_docs * 100) if total_docs > 0 else 0
        
        return {
            'category': category_name,
            'total_documents': total_docs,
            'ai_detected': ai_detected,
            'unlikely_ai': total_docs - ai_detected,
            'ai_percentage': ai_percentage
        }
    except Exception as e:
        print(f"Error processing {category_name}: {e}")
        return None


def main():
    # Base path for results
    results_base_path = Path(__file__).parent.parent / 'data' / 'results'
    
    # Find all pre_pangram_detection_results.csv files
    csv_files = list(results_base_path.glob('*/pre_pangram_detection_results.csv'))
    
    if not csv_files:
        print("No pre_pangram_detection_results.csv files found!")
        return
    
    print(f"Found {len(csv_files)} category files to analyze\n")
    print("=" * 80)
    
    results = []
    
    # Analyze each category
    for csv_file in sorted(csv_files):
        category_name = csv_file.parent.name
        result = analyze_category(csv_file, category_name)
        if result:
            results.append(result)
    
    # Sort results by AI percentage (descending)
    results.sort(key=lambda x: x['ai_percentage'], reverse=True)
    
    # Display results
    print(f"\n{'Category':<15} {'Total':<10} {'AI Detected':<15} {'Unlikely AI':<15} {'AI %':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['category']:<15} {result['total_documents']:<10} "
              f"{result['ai_detected']:<15} {result['unlikely_ai']:<15} "
              f"{result['ai_percentage']:<10.2f}%")
    
    # Calculate overall statistics
    total_all = sum(r['total_documents'] for r in results)
    ai_detected_all = sum(r['ai_detected'] for r in results)
    overall_percentage = (ai_detected_all / total_all * 100) if total_all > 0 else 0
    
    print("=" * 80)
    print(f"{'OVERALL':<15} {total_all:<10} {ai_detected_all:<15} "
          f"{total_all - ai_detected_all:<15} {overall_percentage:<10.2f}%")
    print("=" * 80)
    
    # Save results to CSV
    output_path = results_base_path.parent / 'reports' / 'pre_pangram_ai_detection_summary.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
