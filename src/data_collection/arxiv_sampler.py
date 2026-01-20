"""
Module for sampling papers from the Kaggle arXiv dataset.
Selects random papers from each month between 2020 and 2025.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
import kagglehub
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ArxivSampler:
    """Sample papers from the Kaggle arXiv dataset."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config['output']['directories'].get('raw_data', 'data/raw'))
        self.kaggle_dir = Path('data/kaggle')
        self.kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        # Download or get path to the dataset
        logger.info("Locating arXiv dataset from Kaggle...")
        self.dataset_path = kagglehub.dataset_download("Cornell-University/arxiv")
        logger.info(f"Dataset path: {self.dataset_path}")
        
        self.metadata_file = Path(self.dataset_path) / 'arxiv-metadata-oai-snapshot.json'
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
    
    def parse_arxiv_date(self, date_str: str) -> datetime:
        """
        Parse arXiv date string to datetime object.
        
        Args:
            date_str: Date string in various arXiv formats
            
        Returns:
            datetime object
        """
        # Try various date formats used in arXiv
        formats = [
            '%Y-%m-%d',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%SZ',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If no format works, try to extract just the date part
        try:
            # Extract YYYY-MM-DD from the string
            date_part = date_str.split()[0] if ' ' in date_str else date_str
            return datetime.strptime(date_part, '%Y-%m-%d')
        except:
            logger.warning(f"Could not parse date: {date_str}")
            return None
    
    def get_paper_date(self, paper: dict) -> datetime:
        """
        Extract the submission date from a paper entry.
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            datetime object or None if date cannot be extracted
        """
        # Try to get date from versions (most recent version)
        if 'versions' in paper and paper['versions']:
            try:
                # Get the first version (original submission)
                first_version = paper['versions'][0]
                if 'created' in first_version:
                    return self.parse_arxiv_date(first_version['created'])
            except (IndexError, KeyError, TypeError):
                pass
        
        # Fallback: try to extract from ID (format: YYMM.NNNNN)
        if 'id' in paper:
            arxiv_id = paper['id']
            try:
                # arXiv IDs before 2007 use different format
                # After 2007: YYMM.NNNNN or YYMM.NNNNNV#
                parts = arxiv_id.split('.')
                if len(parts) >= 2:
                    year_month = parts[0]
                    if len(year_month) == 4:
                        year = int('20' + year_month[:2])
                        month = int(year_month[2:])
                        return datetime(year, month, 1)
            except (ValueError, IndexError):
                pass
        
        # Try update_date as last resort
        if 'update_date' in paper:
            return self.parse_arxiv_date(paper['update_date'])
        
        return None
    
    def sample_papers_by_month(
        self,
        start_year: int = 2020,
        end_year: int = 2025,
        samples_per_month: int = 500,
        random_seed: int = 42,
        categories: List[str] = None,
        strict_categories: bool = True
    ) -> pd.DataFrame:
        """
        Sample papers from each month in the specified date range.
        
        Args:
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            samples_per_month: Number of papers to sample per month
            random_seed: Random seed for reproducibility
            categories: List of arXiv categories to filter by (e.g., ['cs.AI', 'cs.LG'])
                       If None, samples from all categories
            strict_categories: If True, only keep papers that belong exclusively to the
                             specified categories (no other categories). If False, keep
                             papers that have at least one matching category.
                             Default is True.
            
        Returns:
            DataFrame with sampled papers
        """
        random.seed(random_seed)
        
        logger.info(f"Reading metadata from {self.metadata_file}")
        logger.info(f"Sampling {samples_per_month} papers per month from {start_year} to {end_year}")
        if categories:
            logger.info(f"Filtering by categories: {categories}")
            logger.info(f"Strict category filtering: {strict_categories}")
        
        # Organize papers by year-month
        papers_by_month = defaultdict(list)
        total_papers = 0
        processed_papers = 0
        skipped_papers = 0
        
        # Read the JSON file line by line (it's a JSONL-like format)
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            pbar = tqdm(f, desc="Processing papers", unit=" papers", unit_scale=True)
            for line_num, line in enumerate(pbar, 1):
                if line_num % 10000 == 0:
                    pbar.set_postfix({"collected": f"{processed_papers:,}"}, refresh=True)
                
                try:
                    paper = json.loads(line)
                    total_papers += 1
                    
                    # Get paper date
                    paper_date = self.get_paper_date(paper)
                    if paper_date is None:
                        skipped_papers += 1
                        continue
                    
                    # Filter by categories if specified
                    if categories:
                        paper_categories = paper.get('categories', '')
                        # Categories are space-separated in arXiv metadata
                        if isinstance(paper_categories, str):
                            paper_cats = paper_categories.split()
                        else:
                            paper_cats = []
                        
                        # Check if any paper category matches our filter
                        # Support both exact match (cs.AI) and prefix match (cs matches cs.AI, cs.LG, etc.)
                        match_found = False
                        for filter_cat in categories:
                            for paper_cat in paper_cats:
                                if paper_cat == filter_cat or paper_cat.startswith(filter_cat + '.'):
                                    match_found = True
                                    break
                            if match_found:
                                break
                        
                        if not match_found:
                            continue
                        
                        everything_matches = True
                        for filter_cat in paper_cats:
                            if not any(filter_cat == cat or filter_cat.startswith(cat + '.') for cat in categories):
                                everything_matches = False
                                break
                        
                        if strict_categories and not everything_matches:
                            continue
                    
                    # Filter by date range
                    if start_year <= paper_date.year <= end_year:
                        # Create year-month key
                        year_month = f"{paper_date.year}-{paper_date.month:02d}"
                        
                        # Store paper with relevant metadata
                        paper_info = {
                            'id': paper.get('id', ''),
                            'title': paper.get('title', ''),
                            'authors': paper.get('authors', ''),
                            'abstract': paper.get('abstract', ''),
                            'categories': paper.get('categories', ''),
                            'doi': paper.get('doi', ''),
                            'journal-ref': paper.get('journal-ref', ''),
                            'comments': paper.get('comments', ''),
                            'update_date': paper.get('update_date', ''),
                            'year_month': year_month,
                            'submission_date': paper_date.strftime('%Y-%m-%d')
                        }
                        
                        papers_by_month[year_month].append(paper_info)
                        processed_papers += 1
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue
            
            pbar.close()
        
        logger.info(f"Total papers read: {total_papers:,}")
        logger.info(f"Papers in target range ({start_year}-{end_year}): {processed_papers:,}")
        logger.info(f"Papers skipped (no date): {skipped_papers:,}")
        logger.info(f"Months with papers: {len(papers_by_month)}")
        
        # Sample from each month
        sampled_papers = []
        sorted_months = sorted(papers_by_month.keys())
        for year_month in tqdm(sorted_months, desc="Sampling by month", unit=" month"):
            papers = papers_by_month[year_month]
            n_papers = len(papers)
            n_samples = min(samples_per_month, n_papers)
            
            # Random sample
            sampled = random.sample(papers, n_samples)
            sampled_papers.extend(sampled)
        
        # Convert to DataFrame
        df = pd.DataFrame(sampled_papers)
        logger.info(f"Total sampled papers: {len(df):,}")
        
        return df
    
    def save_sampled_papers(
        self,
        df: pd.DataFrame,
        output_filename: str = 'arxiv_sampled_papers.csv'
    ):
        """
        Save sampled papers to CSV and JSON files.
        
        Args:
            df: DataFrame with sampled papers
            output_filename: Base filename for output files
        """
        # Save to kaggle directory
        csv_path = self.kaggle_dir / output_filename
        json_path = self.kaggle_dir / output_filename.replace('.csv', '.json')
        
        # Save as CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved sampled papers to {csv_path}")
        
        # Save as JSON
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved sampled papers to {json_path}")
        
        # Print summary statistics
        logger.info(f"\nSummary:")
        logger.info(f"Total papers sampled: {len(df):,}")
        logger.info(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
        logger.info(f"Number of unique months: {df['year_month'].nunique()}")


def sample_arxiv_papers(config: dict):
    """
    Main function to sample papers from the arXiv dataset.
    
    Args:
        config: Configuration dictionary (should contain 'categories' key)
    """
    logger.info("Starting arXiv paper sampling from Kaggle dataset")
    
    try:
        sampler = ArxivSampler(config)
        
        # Get categories from config (passed from command line)
        categories = config.get('categories', None)
        
        # Sample papers
        df = sampler.sample_papers_by_month(
            start_year=2020,
            end_year=2025,
            samples_per_month=500,
            random_seed=42,
            categories=categories
        )
        
        # Create category-specific filename
        if categories:
            category_str = '_'.join(categories).replace('.', '-')
            output_filename = f'arxiv_sampled_papers_{category_str}.csv'
        else:
            output_filename = 'arxiv_sampled_papers.csv'
        
        # Save results
        sampler.save_sampled_papers(df, output_filename=output_filename)
        
        logger.info("arXiv paper sampling completed successfully")
        
    except Exception as e:
        logger.error(f"Error sampling arXiv papers: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # For standalone testing
    import yaml
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sample_arxiv_papers(config)
