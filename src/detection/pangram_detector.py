"""
Pangram API-based AI content detection.
Provides per-paper classification using the Pangram SDK.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict
import pandas as pd
from tqdm import tqdm

from pangram import Pangram

logger = logging.getLogger(__name__)


class PangramDetector:
    """Detect AI-generated content using Pangram SDK."""

    def __init__(self, config: dict):
        if Pangram is None:
            raise ImportError("Pangram SDK not installed. Install with: pip install pangram-sdk")

        self.config = config
        self.pangram_config = config['detection'].get('pangram', {})

        # API configuration
        self.api_key = os.getenv('PANGRAM_API_KEY', self.pangram_config.get('api_key', ''))
        if not self.api_key or self.api_key == '${PANGRAM_API_KEY}':
            raise ValueError("PANGRAM_API_KEY not set. Please set it in environment variables.")

        self.batch_size = self.pangram_config.get('batch_size', 32)
        self.rate_limit_delay = self.pangram_config.get('rate_limit_delay', 1.0)

        # Initialize Pangram client
        try:
            self.client = Pangram(api_key=self.api_key)
            logger.info(f"PangramDetector initialized with batch_size={self.batch_size}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Pangram client: {e}")

    def classify_text(self, text: str) -> Dict:
        """
        Classify a single text using Pangram SDK.

        Args:
            text: Text to classify

        Returns:
            Dictionary with classification results
        """
        try:
            # Use predict_short for abstracts/titles (typically < 512 tokens)
            result = self.client.predict_short(text)
            return self._parse_response(result)

        except Exception as e:
            logger.error(f"Pangram prediction failed: {e}")
            return self._error_response(str(e))

    def classify_texts_batch(self, texts: Dict[str, str]) -> Dict[str, Dict]:
        """
        Classify multiple texts using Pangram SDK batch processing.

        Args:
            texts: Dictionary mapping paper IDs to text content

        Returns:
            Dictionary mapping paper IDs to classification results
        """
        results = {}
        paper_ids = list(texts.keys())

        # Process in batches
        for i in tqdm(range(0, len(paper_ids), self.batch_size), desc="Classifying with Pangram"):
            batch_ids = paper_ids[i:i + self.batch_size]
            batch_texts = [texts[pid] for pid in batch_ids]

            try:
                # Use SDK's batch_predict method
                batch_results = self.client.batch_predict(batch_texts)

                # Map results back to paper IDs
                for paper_id, result in zip(batch_ids, batch_results):
                    results[paper_id] = self._parse_response(result)

            except Exception as e:
                logger.error(f"Batch prediction failed: {e}, falling back to individual requests")
                # Fall back to individual requests
                for paper_id in batch_ids:
                    try:
                        result = self.client.predict_short(texts[paper_id])
                        results[paper_id] = self._parse_response(result)
                    except Exception as e2:
                        logger.error(f"Individual prediction failed for {paper_id}: {e2}")
                        results[paper_id] = self._error_response(str(e2))

            # Rate limiting between batches
            if i + self.batch_size < len(paper_ids):
                time.sleep(self.rate_limit_delay)

        return results

    def _parse_response(self, response) -> Dict:
        """
        Parse Pangram SDK response and extract relevant fields.

        Args:
            response: Response from Pangram SDK (can be dict or object with attributes)

        Returns:
            Parsed classification result
        """
        try:
            # SDK returns dict with 'ai_likelihood' field (0=human, 1=AI)
            ai_likelihood = response.get('ai_likelihood', -1)

            text = response.get('text', '')
            prediction = response.get('prediction', {})
            llm_prediction = response.get('llm_prediction', {})
            
            result = {
                'ai_likelihood': ai_likelihood,
                'text': text,
                'prediction': prediction,
                'llm_prediction': llm_prediction
            }

            return result

        except Exception as e:
            logger.error(f"Error parsing Pangram response: {e}")
            return self._error_response(str(e))

    def _error_response(self, error_msg: str) -> Dict:
        """Create error response."""
        return {
            'pangram_label': None,
            'pangram_confidence': None,
            'pangram_ai_likelihood': None,
            'pangram_error': error_msg
        }


def detect_pangram_content(config: dict):
    """
    Detect AI-generated content using Pangram API for all papers.

    Args:
        config: Configuration dictionary
    """
    logger.info("Starting Pangram AI content detection")

    # Load paper data
    processed_dir = Path(config['output']['directories']['processed_data'])
    metadata_path = processed_dir / 'paper_metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"paper_metadata.csv not found in {processed_dir}. Run preprocess stage first."
        )
    text_df = pd.read_csv(metadata_path)

    # Ensure required columns exist
    if 'period' not in text_df.columns:
        text_df['period'] = 'unknown'
    else:
        text_df['period'] = text_df['period'].fillna('unknown')

    # Load classification results for paper_type
    try:
        class_df = pd.read_csv(processed_dir / 'paper_classifications.csv')
        text_df = pd.merge(text_df, class_df[['arxiv_id', 'predicted_type']], on='arxiv_id', how='left')
        text_df['paper_type'] = text_df['predicted_type']
        text_df.drop(columns=['predicted_type'], inplace=True)
        logger.info("Merged classification results (paper_type information)")
    except Exception as e:
        logger.warning(f"Could not load classifications: {e}")
        text_df['paper_type'] = 'unknown'

    # Extract year and month from submission_date
    text_df['year'] = pd.to_datetime(text_df['submission_date'], errors='coerce').dt.year
    text_df['month'] = pd.to_datetime(text_df['submission_date'], errors='coerce').dt.month

    # Construct full_text from title + abstract
    logger.info("Constructing text from title and abstract")
    text_df['full_text'] = (
        text_df['title'].fillna('').astype(str) + '\n\n' +
        text_df['abstract'].fillna('').astype(str)
    )

    # Initialize Pangram detector
    detector = PangramDetector(config)

    # Sample 100 random papers per month and year
    sampled_dfs = []
    for (year, month), group in text_df.groupby(['year', 'month']):
        sample_size = min(100, len(group))
        sampled = group.sample(n=sample_size, random_state=42)
        sampled_dfs.append(sampled)
        logger.info(f"Sampled {sample_size} papers from {year}-{month:02d} (out of {len(group)} available)")

    text_df = pd.concat(sampled_dfs, ignore_index=True)
    
    categories = config['categories'][0]
    # when running the analysis on the subcategories only consider years from 2023 onward
    if '.' in categories:
        # remove years earlier than 2023
        text_df = text_df[text_df['year'] >= 2023]
        
    logger.info(f"Total sampled papers for Pangram detection: {len(text_df)}")

    # Prepare texts for batch classification
    logger.info("Preparing texts for batch classification")
    texts_dict = {}
    id_to_row = {}

    for ind, row in text_df.iterrows():
        arxiv_id = row['arxiv_id']
        full_text = row.get('full_text', '')
        if full_text and not pd.isna(full_text):
            texts_dict[arxiv_id] = full_text
            id_to_row[arxiv_id] = row

    logger.info(f"Prepared {len(texts_dict)} texts for batch classification")

    # Classify all texts using batch processing
    pangram_results = detector.classify_texts_batch(texts_dict)

    # Construct detection results
    detection_results = []
    for arxiv_id, pangram_result in pangram_results.items():
        row = id_to_row[arxiv_id]
        result = {
            'arxiv_id': arxiv_id,
            'paper_type': row.get('paper_type', 'unknown'),
            'period': row.get('period', 'unknown'),
            'year': row.get('year', None) if not pd.isna(row.get('year')) else None,
            'month': row.get('month', None) if not pd.isna(row.get('month')) else None,
            'pangram_prediction': pangram_result,
        }
        detection_results.append(result)
        
    # Save results to pangram-specific directory
    df_results = pd.DataFrame(detection_results)

    # Update output directories to use pangram suffix
    results_dir = Path(config['output']['directories']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(results_dir / 'pangram_detection_results.csv', index=False)
    df_results.to_json(results_dir / 'pangram_detection_results.json', orient='records', indent=2)
    logger.info(f"Saved Pangram detection results to {results_dir}")

    # Summary statistics
    # Extract ai_likelihood from pangram_prediction dictionaries
    df_results['ai_likelihood'] = df_results['pangram_prediction'].apply(
        lambda x: x.get('ai_likelihood', -1) if isinstance(x, dict) else -1
    )
    valid_classifications = (df_results['ai_likelihood'] >= 0).sum()
    logger.info(f"Successfully classified {valid_classifications}/{len(df_results)} papers")

    if valid_classifications > 0:
        # Threshold at 0.5 to determine AI-generated (ai_likelihood ranges from 0=human to 1=AI)
        ai_generated_count = (df_results['ai_likelihood'] >= 0.5).sum()
        logger.info(f"Papers classified as AI-generated: {ai_generated_count} ({ai_generated_count/valid_classifications*100:.1f}%)")
        logger.info(f"Mean AI likelihood: {df_results[df_results['ai_likelihood'] >= 0]['ai_likelihood'].mean():.3f}")

    return df_results
