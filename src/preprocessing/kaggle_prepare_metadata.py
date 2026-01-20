"""Prepare metadata for classification from Kaggle arXiv sample.

This module converts the sampled Kaggle arXiv CSV produced by
`src/data_collection/arxiv_sampler.py` into the expected
`processed_data/paper_metadata.csv` used by the downstream classification
stage. Since we do not have PDFs in this pathway, we omit text extraction.
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _determine_period(submission_date: str, config: dict) -> str:
    """Classify a paper into pre/post LLM periods based on submission date.

    Args:
        submission_date: Date string (YYYY-MM-DD)
        config: Global configuration dict.
    Returns:
        'pre_llm' or 'post_llm'
    """
    try:
        dt = datetime.strptime(submission_date, "%Y-%m-%d")
    except Exception:
        return "unknown"
    pre_end = datetime.strptime(config['data_collection']['time_periods']['pre_llm_end'], "%Y-%m-%d")
    post_start = datetime.strptime(config['data_collection']['time_periods']['post_llm_start'], "%Y-%m-%d")
    if dt <= pre_end:
        return "pre_llm"
    if dt >= post_start:
        return "post_llm"
    return "unknown"


def prepare_kaggle_metadata(config: dict):
    """Prepare metadata for Kaggle arXiv sampled papers.

    Reads `data/kaggle/arxiv_sampled_papers.csv` (or category-specific file) and writes
    `data/processed/paper_metadata.csv` with required columns:
      - arxiv_id
      - title
      - abstract
      - paper_type (set to 'unknown')
      - period (pre_llm / post_llm / unknown)
      - submission_date
      - categories
      - doi
      - comments
    """
    logger.info("Preparing Kaggle arXiv sampled metadata for classification")

    processed_dir = Path(config['output']['directories']['processed_data'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Determine input filename based on categories in config
    categories = config.get('categories', None)
    if categories:
        category_str = '_'.join(categories).replace('.', '-')
        sampled_path = Path(f'data/kaggle/arxiv_sampled_papers_{category_str}.csv')
    else:
        sampled_path = Path('data/kaggle/arxiv_sampled_papers.csv')
    if not sampled_path.exists():
        raise FileNotFoundError(f"Sampled Kaggle arXiv file not found: {sampled_path}")

    df = pd.read_csv(sampled_path)

    # Normalize / rename columns
    rename_map = {
        'id': 'arxiv_id',
        'journal-ref': 'journal_ref'
    }
    df = df.rename(columns=rename_map)

    # Derive period
    df['period'] = df['submission_date'].apply(lambda d: _determine_period(str(d), config))

    # paper_type unknown in this pathway (will be predicted later)
    df['paper_type'] = 'unknown'

    # Select and order columns for downstream consistency
    cols = [
        'arxiv_id', 'title', 'authors', 'abstract', 'paper_type', 'period', 'submission_date', 'update_date', 'year_month',
        'categories', 'doi', 'journal_ref', 'comments'
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ''
    df_out = df[cols].copy()

    out_csv = processed_dir / 'paper_metadata.csv'
    df_out.to_csv(out_csv, index=False)
    df_out.to_json(processed_dir / 'paper_metadata.json', orient='records', indent=2)

    logger.info(f"Saved prepared metadata: {len(df_out)} papers -> {out_csv}")
    logger.info(f"Period distribution: {df_out['period'].value_counts().to_dict()}")
    return df_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    cfg = yaml.safe_load(open('config/config.yaml'))
    prepare_kaggle_metadata(cfg)
