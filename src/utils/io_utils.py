"""
I/O utility functions for loading and enriching data.

This module provides reusable functions for common data loading and enrichment
operations used throughout the pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def enrich_with_year(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Merge detection results with metadata to add publication year.

    This function loads paper metadata and extracts the publication year,
    using fallback logic (published -> updated -> submission_date) to handle
    missing values.

    Args:
        df: DataFrame containing detection results with 'arxiv_id' column
        config: Configuration dictionary containing output directory paths

    Returns:
        DataFrame enriched with 'year' column containing publication year

    Note:
        If 'year' column already exists with non-null values, enrichment is skipped.
        Expects metadata file at: {processed_data}/paper_metadata.csv
    """
    # If year already present and has non-null values, skip enrichment
    if 'year' in df.columns and df['year'].notna().any():
        logger.debug("Year column already present, skipping enrichment")
        return df

    try:
        processed_dir = Path(config['output']['directories']['processed_data'])
        metadata_file = processed_dir / 'paper_metadata.csv'

        # Try to load only required columns first for efficiency
        # Force arxiv_id to be string to avoid float conversion issues
        try:
            meta = pd.read_csv(
                metadata_file,
                usecols=['arxiv_id', 'published', 'updated', 'submission_date'],
                dtype={'arxiv_id': str}
            )
        except Exception:
            # Fallback: load all columns if specific columns don't exist
            meta = pd.read_csv(metadata_file, dtype={'arxiv_id': str})
            # Keep only expected columns if they exist
            keep = [c for c in ['arxiv_id', 'published', 'updated', 'submission_date']
                   if c in meta.columns]
            meta = meta[keep]

        # Fallback logic for published date: published -> updated -> submission_date
        if 'published' not in meta.columns and 'updated' in meta.columns:
            meta['published'] = meta['updated']
        elif 'published' not in meta.columns and 'submission_date' in meta.columns:
            meta['published'] = meta['submission_date']
        elif 'published' in meta.columns and 'updated' in meta.columns:
            meta['published'] = meta['published'].fillna(meta['updated'])
        elif 'published' in meta.columns and 'submission_date' in meta.columns:
            meta['published'] = meta['published'].fillna(meta['submission_date'])

        # Extract year from published date
        if 'published' in meta.columns:
            meta['year'] = pd.to_datetime(meta['published'], errors='coerce').dt.year
        else:
            logger.warning("No date column found in metadata, year will be NaN")
            meta['year'] = np.nan

        # Deduplicate metadata by arxiv_id (keep first occurrence)
        meta = meta.drop_duplicates(subset=['arxiv_id'])

        # Merge with detection results
        if 'arxiv_id' in df.columns:
            # Ensure both arxiv_id columns have the same dtype (string)
            df['arxiv_id'] = df['arxiv_id'].astype(str)
            meta['arxiv_id'] = meta['arxiv_id'].astype(str)

            df = df.merge(meta[['arxiv_id', 'year']], on='arxiv_id', how='left')
            logger.info(f"Enriched {len(df)} papers with publication year")
        else:
            # If arxiv_id missing, cannot enrich
            logger.warning("No 'arxiv_id' column in DataFrame, cannot enrich with year")
            df['year'] = np.nan

        return df

    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_file}")
        df['year'] = np.nan
        return df
    except Exception as e:
        logger.error(f"Error enriching with year: {e}")
        df['year'] = np.nan
        return df


def load_metadata(processed_dir: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load paper metadata from the processed data directory.

    Args:
        processed_dir: Path to the processed data directory
        columns: Optional list of specific columns to load for efficiency

    Returns:
        DataFrame containing paper metadata

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        pd.errors.ParserError: If CSV file is malformed
    """
    metadata_file = processed_dir / 'paper_metadata.csv'

    try:
        if columns:
            df = pd.read_csv(metadata_file, usecols=columns)
        else:
            df = pd.read_csv(metadata_file)

        logger.info(f"Loaded metadata for {len(df)} papers from {metadata_file}")
        return df

    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_file}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing metadata CSV: {e}")
        raise
